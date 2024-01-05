#![feature(adt_const_params)]
#![feature(split_array)]
#![feature(exclusive_range_pattern)]
#![feature(variant_count)]
#![feature(array_methods)]
#![feature(generic_arg_infer)]
#![feature(int_roundings)]
#![feature(lazy_cell)]

use std::cell::LazyCell;
use std::collections::{BTreeSet, BTreeMap};
use std::f64::{EPSILON, INFINITY};
use std::f64::consts::TAU;
use std::ffi::OsStr;
use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{Ordering, AtomicU8, AtomicI8, AtomicUsize};
use std::{vec, fs};

use array_math::ArrayOps;
use num::rational::Ratio;
use parameters::{SelectionParameters, EnvelopeParameters, OscillatorParameters, RoutingParameters, LFOParameters, FilterParameters};
use real_time_fir_iir_filters::Filter;
use serde_scala::{Scale, SerdeScalaError};
use tuning::Tuning;
use vst::channels::{ChannelInfo, SpeakerArrangementType, StereoChannel, StereoConfig};
use vst::{prelude::*, plugin_main};

use real_time_fir_iir_filters::iir::{first::FirstOrderFilter, second::SecondOrderFilter};
use self::envelope::ADSREnvelope;
use self::lfo::LFO;
use self::oscillator::Oscillator;
use self::parameters::{MekkaSynthParameters, Control};
use self::waveform::Waveform;

moddef::moddef!(
    flat(pub) mod {
        parameters,
        waveform,
        oscillator,
        lfo,
        tuning,
        envelope
    }
);

pub const POLY_MAX: usize = 7;
pub const POLY_MIN: usize = 1;
pub const DETUNE_MAX: f64 = 2.0;
pub const DETUNE_MIN: f64 = -2.0;
pub const PORTAMENTO_MAX: f64 = 10.0;
pub const PORTAMENTO_MIN: f64 = 0.0005 - EPSILON;
pub const ATTACK_MAX: f64 = 10.0;
pub const ATTACK_MIN: f64 = 0.0005 - EPSILON;
pub const DECAY_MAX: f64 = 10.0;
pub const DECAY_MIN: f64 = 0.0005 - EPSILON;
pub const RELEASE_MAX: f64 = 10.0;
pub const RELEASE_MIN: f64 = 0.0005 - EPSILON;
pub const FILTER_FREQUENCY_MIN: f64 = 20.0;
pub const FILTER_RESONANCE_MAX: f64 = 20.0;
pub const FILTER_RESONANCE_MIN: f64 = 0.0;
pub const ENVELOPE_VCO_MAX: f64 = 12.0;
pub const ENVELOPE_VCO_MIN: f64 = -12.0;
pub const LFO_FREQUENCY_MAX: f64 = 100.0;
pub const LFO_FREQUENCY_MIN: f64 = 0.001;

pub const OSCILLATOR_COUNT: usize = 3;
pub const LFO_COUNT: usize = 3;
pub const ENVELOPE_COUNT: usize = 3;

const FILTER_CHANGE: f64 = 0.2;
const PITCH_CHANGE: f64 = 0.4;

const TOTAL_CHANNEL_COUNT: usize = CHANNEL_COUNT + POLY_MAX + 1;

const GAIN: f64 = 0.6;

const KEYS: [&str; 12] = [
    "C",
    "G",
    "D",
    "A",
    "E",
    "B",
    "F#/Gb",
    "Db",
    "Ab",
    "Eb",
    "Bb",
    "F"
];
const DEFAULT_SCALE: &str = "12edo";

struct MekkaSynthPlugin
{
    host: HostCallback,
    pub param: Arc<MekkaSynthParameters>,
    lfo: [LFO; LFO_COUNT],
    osc: [[Oscillator; OSCILLATOR_COUNT]; POLY_MAX],
    env: [[ADSREnvelope; ENVELOPE_COUNT]; POLY_MAX],
    tuning_name: Option<String>,
    tuning: Tuning,
    notes_held: [(i32, f64); POLY_MAX],
    press_order: Vec<usize>,
    repress: Vec<(i32, f64)>,
    pressed: Vec<(i32, f64)>,
    pitch: f64,
    pitch_control: f64,
    filter_env: [[FirstOrderFilter<f64>; ENVELOPE_COUNT]; POLY_MAX],
    filter_lfo: [FirstOrderFilter<f64>; LFO_COUNT],
    filter: [[SecondOrderFilter<f64>; POLY_MAX]; CHANNEL_COUNT],
    rate: f64
}

const CHANNEL_COUNT: usize = 2;

const SCALES_DIR: &str = "Scales";

impl MekkaSynthPlugin
{
    fn fetch_scale<'a>(scales: &'a BTreeMap<String, RwLock<Option<Scale>>>, scale_name: &String) -> Result<Option<Scale>, SerdeScalaError>
    {
        if let Some(scale) = scales.get(scale_name)
        {
            {
                let scale = scale.read().unwrap();
                if scale.is_some()
                {
                    return Ok(scale.clone())
                }
            }

            let path = format!("{}\\{}.scl", SCALES_DIR, scale_name);
            
            let bytes = fs::read(path)?;
            let contents = String::from_utf8_lossy(&bytes);
            *scale.write().unwrap() = Some(contents.parse()?);
            
            return Ok(scale.read().unwrap().clone())
        }

        Ok(None)
    }

    fn fetch_scales() -> Result<BTreeMap<String, RwLock<Option<Scale>>>, SerdeScalaError>
    {
        let mut scales = BTreeMap::new();

        let cd = fs::read_dir(SCALES_DIR).or_else(|_| {
            fs::create_dir(SCALES_DIR)?;
            fs::read_dir(SCALES_DIR)
        })?;

        for entry in cd
        {
            let entry = entry?;
            let file_name = entry.file_name();
            if let Some(file_name) = file_name.to_str()
            {
                if entry.file_type()?.is_file() && file_name.ends_with(".scl")
                {
                    let scale_name = file_name.split_at(file_name.len() - ".scl".len()).0;

                    scales.insert(scale_name.to_string(), RwLock::new(None));
                }
            }
        }

        Ok(scales)
    }

    fn note_offset_with_key(key: u8) -> i32
    {
        match key % 12
        {
            0 => 0, // C
            1 => -5, // G
            2 => 2, // D
            3 => -3, // A
            4 => 4, // E
            5 => -1, // B
            6 => 6, // F#
            7 => 1, // Db
            8 => -4, // Ab
            9 => 3, // Eb
            10 => -2, // Bb
            11 => 5, // F
            _ => panic!()
        }
    }

    fn get_closest_osc(&self, free_osc: &[usize], note: f64) -> Option<usize>
    {
        let mut i = None;
        let mut dn = INFINITY;
        for i_ in 0..free_osc.len()
        {
            let dn_ = (0..OSCILLATOR_COUNT)
                .map(|n| (self.osc[free_osc[i_]][n].note - note).abs())
                .reduce(|a, b| a.min(b))
                .unwrap_or(INFINITY);
            if dn_ < dn
            {
                i = Some(i_);
                dn = dn_;
            }
        }
        return i;
    }

    fn press_note(&mut self, note: i32, detune: f64)
    {
        let poly = self.param.polyphony.load(Ordering::Relaxed) as usize;
        self.unpress_note(note);
        self.pressed.push((note, detune));

        let note_f = self.tuning.note(note) + Self::note_offset_with_key(self.param.key.load(Ordering::Relaxed)) as f64 + detune;

        let mut not_held = vec![];
        for i in 0..poly
        {
            if !self.env[i].iter().any(|e| e.is_held())
            {
                not_held.push(i);
            }
        }
        match self.get_closest_osc(&not_held, note_f)
        {
            Some(i) => {
                let i = not_held[i];
                self.notes_held[i] = (note, detune);
                for j in 0..ENVELOPE_COUNT
                {
                    self.env[i][j].i = Some(0);
                    self.env[i][j].h = None;
                }
                self.press_order = [
                    vec![i],
                    self.press_order.iter()
                        .filter(|&&i_| i_ != i)
                        .map(|&i| i)
                        .collect()
                ].concat();
                while self.press_order.len() > poly
                {
                    self.press_order.pop();
                }
                return
            }
            None => ()
        }
        let mut held = vec![];
        for &i in self.press_order.iter()
        {
            if i < poly
            {
                held.push(i);
            }
        }
        match self.get_closest_osc(&held, note_f)
        {
            Some(i) => {
                let i = held[i];
                self.repress = [
                    vec![self.notes_held[i]],
                    self.repress.iter()
                        .filter(|&&n| n != self.notes_held[i])
                        .map(|&n| n)
                        .collect()
                ].concat();
                self.notes_held[i] = (note, detune);
                for j in 0..ENVELOPE_COUNT
                {
                    self.env[i][j].i = Some(0);
                    self.env[i][j].h = None;
                }
                self.press_order = [
                    vec![i],
                    self.press_order.iter()
                        .filter(|&&i_| i_ != i)
                        .map(|&i| i)
                        .collect()
                ].concat();
                while self.press_order.len() > poly
                {
                    self.press_order.pop();
                }
            },
            None => ()
        }
    }

    fn unpress_note(&mut self, note: i32)
    {
        self.pressed = self.pressed.iter()
            .filter(|&&n| n.0 != note)
            .map(|&n| n)
            .collect();
        for i in 0..POLY_MAX
        {
            if self.env[i].iter().any(|e| e.is_held()) && self.notes_held[i].0 == note
            {
                for j in 0..ENVELOPE_COUNT
                {
                    self.env[i][j].h = self.env[i][j].i
                }
            }
        }
        loop
        {
            match self.repress.last()
            {
                Some(&n) => {
                    self.repress.pop();
                    if self.pressed.contains(&n)
                    {
                        self.press_note(n.0, n.1); return;
                    }
                },
                _ => return
            }
        }
    }
}

impl Plugin for MekkaSynthPlugin
{
    fn new(host: HostCallback) -> Self
    where
        Self: Sized
    {
        let filter_frequency_max = 22050.0;
        let zeta = 0.5f64.sqrt();
        let waveform = Waveform::Sine;
        let waveform_lfo = Waveform::Triangle;
        let scales = MekkaSynthPlugin::fetch_scales().unwrap();
        let tuning = scales.keys()
            .enumerate()
            .filter(|(_, name)| *name == DEFAULT_SCALE)
            .map(|(i, _)| i)
            .next()
            .unwrap();
        Self {
            host,
            param: Arc::new(MekkaSynthParameters {
                select: SelectionParameters {
                    osc: AtomicU8::new(0),
                    lfo: AtomicU8::new(0),
                    envelope: AtomicU8::new(0),
                    vclfo: AtomicU8::new(0)
                },
                osc: ArrayOps::fill(|i| OscillatorParameters {
                    waveform: AtomicU8::new(waveform as u8),
                    duty_cycle: AtomicFloat::new(0.5),
                    detune: AtomicFloat::new(0.0),
                    octave: AtomicI8::new(0),
                    mix: AtomicFloat::new(if i == 0 {1.0} else {0.0}),
                    pan: AtomicFloat::new(0.5)
                }),
                envelope: ArrayOps::fill(|i| EnvelopeParameters {
                    attack: AtomicFloat::new(ATTACK_MIN as f32),
                    decay: AtomicFloat::new(DECAY_MIN as f32),
                    sustain: AtomicFloat::new(1.0),
                    release: AtomicFloat::new(RELEASE_MIN as f32),
                    routing: RoutingParameters {
                        vca: AtomicFloat::new(if i == 0 {1.0} else {0.0}),
                        vcf: AtomicFloat::new(0.0),
                        vco: AtomicFloat::new(0.0),
                        vclfo: [(); _].map(|()| AtomicFloat::new(0.0)),
                    }
                }),
                lfo: [(); _].map(|()| LFOParameters {
                    waveform: AtomicU8::new(waveform_lfo as u8),
                    duty_cycle: AtomicFloat::new(0.5),
                    frequency: AtomicFloat::new(1.0),
                    routing: RoutingParameters {
                        vca: AtomicFloat::new(0.0),
                        vcf: AtomicFloat::new(0.0),
                        vco: AtomicFloat::new(0.0),
                        vclfo: [(); _].map(|()| AtomicFloat::new(0.0)),
                    },
                }),
                filter: FilterParameters {
                    type_: AtomicU8::new(0),
                    frequency: AtomicFloat::new(filter_frequency_max as f32),
                    resonance: AtomicFloat::new(zeta as f32),
                },
                key: AtomicU8::new(0),
                tuning: AtomicUsize::new(tuning),
                polyphony: AtomicU8::new(4),
                portamento: AtomicFloat::new(PORTAMENTO_MIN as f32),
                mix_mode: AtomicU8::new(0),
                volume: AtomicFloat::new(0.5),
                filter_frequency_max: AtomicFloat::new(filter_frequency_max as f32),
                scales: Arc::new(scales)
            }),
            lfo: [LFO::new(TAU*1.0, waveform_lfo); LFO_COUNT],
            osc: [(); _].map(|()| [(); _].map(|()| Oscillator::new(40.0, waveform, 0.5))),
            env: [[ADSREnvelope::new(); ENVELOPE_COUNT]; POLY_MAX],
            tuning_name: None,
            tuning: Tuning::edo(12, Ratio::new(2, 1)),
            notes_held: [(49, 0.0); POLY_MAX],
            press_order: vec![],
            repress: vec![],
            pressed: vec![],
            pitch: 0.0,
            pitch_control: 0.0,
            filter_env: [[FirstOrderFilter::new(TAU*100.0); ENVELOPE_COUNT]; POLY_MAX],
            filter_lfo: [FirstOrderFilter::new(TAU*100.0); LFO_COUNT],
            filter: [[SecondOrderFilter::new(TAU*filter_frequency_max, zeta); POLY_MAX]; CHANNEL_COUNT],
            rate: 44100.0
        }
    }

    fn can_do(&self, can_do: CanDo) -> Supported
    {
        match can_do
        {
            CanDo::SendEvents => Supported::Yes,
            CanDo::SendMidiEvent => Supported::No,
            CanDo::ReceiveEvents => Supported::Yes,
            CanDo::ReceiveMidiEvent => Supported::Yes,
            CanDo::ReceiveTimeInfo => Supported::Yes,
            CanDo::Offline => Supported::Yes,
            CanDo::MidiProgramNames => Supported::No,
            CanDo::Bypass => Supported::No,
            CanDo::ReceiveSysExEvent => Supported::No,
        
            //Bitwig specific?
            CanDo::MidiSingleNoteTuningChange => Supported::No,
            CanDo::MidiKeyBasedInstrumentControl => Supported::Yes,
            CanDo::Other(_) => Supported::Maybe
        }
    }

    fn get_info(&self) -> Info
    {
        Info {
            name: "Mekka Synth".to_string(),
            vendor: "Soma FX".to_string(),
            presets: 0,
            parameters: Control::CONTROLS.len() as i32,
            inputs: 0,
            outputs: CHANNEL_COUNT as i32,
            midi_inputs: 1,
            midi_outputs: 0,
            unique_id: 7574754,
            version: 1,
            category: Category::Synth,
            initial_delay: 0,
            preset_chunks: false,
            f64_precision: true,
            silent_when_stopped: true,
            ..Default::default()
        }
    }

    fn get_output_info(&self, output: i32) -> ChannelInfo
    {
        const N: usize = CHANNEL_COUNT + 1;
        ChannelInfo::new(
            match output as usize
            {
                CHANNEL_COUNT => "Total Envelope".to_string(),
                N..TOTAL_CHANNEL_COUNT => format!("Envelope {}", output - CHANNEL_COUNT as i32 - 1),
                _ => format!("Output channel {}", output)
            },
            Some(format!("Out {}", output)),
            output < TOTAL_CHANNEL_COUNT as i32,
            match output as usize
            {
                0 => Some(SpeakerArrangementType::Stereo(StereoConfig::L_R, StereoChannel::Left)),
                1 => Some(SpeakerArrangementType::Stereo(StereoConfig::L_R, StereoChannel::Right)),
                _ => Some(SpeakerArrangementType::Mono)
            }
        )
    }

    fn set_sample_rate(&mut self, rate: f32)
    {
        self.rate = rate as f64;
    }

    fn init(&mut self)
    {

    }

    fn process_events(&mut self, events: &vst::api::Events)
    {
        for event in events.events()
        {
            match event
            {
                vst::event::Event::Midi(midi_event) => {
                    let detune = (midi_event.detune as i32 - ((u8::MAX >> 1)/2) as i32) as f64/100.0;
                    match midi_event.data[0] >> 4
                    {
                        8 => self.unpress_note(midi_event.data[1] as i32 - 20),
                        9 => self.press_note(midi_event.data[1] as i32 - 20, detune),
                        11 => {
                            let c = midi_event.data[1] as i32;
                            let x = midi_event.data[2] as f32/(u8::MAX >> 1) as f32;
                            if c < Control::CONTROLS.len() as i32
                            {
                                self.param.set_parameter(c, x)
                            }
                        }
                        14 => {
                            let p = ((midi_event.data[2] as u32) << 7) + midi_event.data[1] as u32;
                            self.pitch_control = (p as i32 - ((u16::MAX >> 2)/2) as i32) as f64/(u16::MAX >> 2) as f64*24.0
                        }
                        _ => ()
                    }
                }
                _ => ()
            }
        }
    }

    fn process(&mut self, buffer: &mut AudioBuffer<f32>)
    {
        if let Some(tuning_name) = self.param.tuning_name()
        {
            if self.tuning_name.as_ref() != Some(tuning_name)
            {
                if let Some(tuning) = Self::fetch_scale(&self.param.scales, tuning_name).unwrap()
                {
                    self.tuning = Tuning::from_scale(tuning);
                    self.tuning_name = Some(tuning_name.clone());
                }
            }
        }

        let filter_frequency_max = self.rate/2.0;
        self.param.filter_frequency_max.set(filter_frequency_max as f32);

        let poly = self.param.polyphony.load(Ordering::Relaxed) as usize;
        let vol = self.param.volume.get();
        let change_f = (-1.0/self.param.portamento.get() as f64/self.rate).exp();

        let filter_type = self.param.filter.type_.load(Ordering::Relaxed) as usize;

        let waveform = self.param.osc.each_ref()
            .map(|osc| osc.waveform());
        let duty_cycle = self.param.osc.each_ref()
            .map(|osc| osc.duty_cycle.get());
        let mix = self.param.osc.each_ref()
            .map(|osc| osc.mix.get());
        let pan = self.param.osc.each_ref()
            .map(|osc| osc.pan.get());
        let detune = self.param.osc.each_ref()
            .map(|osc| osc.detune.get() as f64 + self.pitch + 12.0*osc.octave.load(Ordering::Relaxed) as f64);
        
        let env_vcf = self.param.envelope.each_ref()
            .map(|env| env.routing.vcf.get());
        let env_vco = self.param.envelope.each_ref()
            .map(|env| env.routing.vco.get());
        let env_vca = self.param.envelope.each_ref()
            .map(|env| env.routing.vca.get());
        let env_vclfo = self.param.envelope.each_ref()
            .map(|env| env.routing.vclfo.each_ref().map(|vclfo| vclfo.get()));

        let lfo_vcf = self.param.lfo.each_ref()
            .map(|lfo| lfo.routing.vcf.get());
        let lfo_vco = self.param.lfo.each_ref()
            .map(|lfo| lfo.routing.vco.get());
        let lfo_vca = self.param.lfo.each_ref()
            .map(|lfo| lfo.routing.vca.get());
        let lfo_vclfo = self.param.lfo.each_ref()
            .map(|lfo| lfo.routing.vclfo.each_ref().map(|vclfo| vclfo.get()));

        let mix_mode = self.param.mix_mode.load(Ordering::Relaxed);

        let lfo_dtc = self.param.lfo.each_ref()
            .map(|lfo| lfo.duty_cycle.get());

        self.pitch = PITCH_CHANGE*self.pitch_control + (1.0 - PITCH_CHANGE)*self.pitch;
        for (lfo, params) in self.lfo.each_mut()
            .zip(self.param.lfo.each_ref())
        {
            lfo.waveform = params.waveform();
        }
        
        for i in 0..poly
        {
            for c in 0..CHANNEL_COUNT
            {
                self.filter[c][i].zeta = FILTER_CHANGE*0.5/(self.param.filter.resonance.get() as f64 + EPSILON) + (1.0 - FILTER_CHANGE)*self.filter[c][i].zeta;
            }
        }

        let samples = buffer.samples();
        let (_, mut outputs) = buffer.split();
        
        {
            for (osc, env) in self.osc.iter_mut()
                .zip(self.env.iter_mut())
            {
                for (osc, (waveform, duty_cycle)) in osc.each_mut()
                    .zip(waveform.zip(duty_cycle))
                {
                    osc.set_waveform(waveform, duty_cycle as f64);
                }

                for (env, params) in env.each_mut()
                    .zip(self.param.envelope.each_ref())
                {
                    env.set_parameters(params)
                }
            }
        }
        let mut g: [[Vec<f64>; ENVELOPE_COUNT]; POLY_MAX] = [(); _].map(|()| [(); _].map(|()| vec![0.0; samples]));

        let y: Vec<[f64; CHANNEL_COUNT]> = (0..samples).map(|s| {
            let lfo: [f64; LFO_COUNT] = ArrayOps::fill(|n| 
                self.filter_lfo[n].filter(self.rate, self.lfo[n].next(self.rate, lfo_dtc[n] as f64))[0]
            );

            for i in 0..poly
            {
                let (note, detune_note) = self.notes_held[i];
                
                for n in 0..ENVELOPE_COUNT
                {
                    g[i][n][s] = self.filter_env[i][n].filter(self.rate, self.env[i][n].next(self.rate))[0];
                }
                for n in 0..OSCILLATOR_COUNT
                {
                    let detune = detune[n]
                        + (0..ENVELOPE_COUNT).map(|m| (1.0 - g[i][m][s])*env_vco[m] as f64).reduce(|a, b| a + b).unwrap_or(0.0)
                        + (0..LFO_COUNT).map(|m| lfo[m].max(-1.0).min(1.0)*lfo_vco[m] as f64).reduce(|a, b| a + b).unwrap_or(0.0);
                
                    self.osc[i][n].note = (self.tuning.note(note) + Self::note_offset_with_key(self.param.key.load(Ordering::Relaxed)) as f64 + detune_note)*(1.0 - change_f)
                        + (self.osc[i][n].note - detune)*change_f
                        + detune
                }

                let f0 = self.param.filter.frequency.get() as f64;

                let a0 = (0..ENVELOPE_COUNT)
                    .map(|m| env_vcf[m].abs() as f64)
                    .reduce(|a, b| a + b)
                    .unwrap_or(0.0)
                    + (0..LFO_COUNT)
                    .map(|m| lfo_vcf[m].abs() as f64)
                    .reduce(|a, b| a + b)
                    .unwrap_or(0.0);
                let f = if a0 == 0.0 || a0 == -0.0
                {
                    f0
                }
                else
                {
                    env_vcf.iter()
                        .map(|&env_vcf| env_vcf as f64)
                        .zip(g[i].iter())
                        .map(|(env_vcf, g)| env_vcf.abs()/a0*if env_vcf >= 0.0
                    {
                        (f0.log2()*((1.0 - env_vcf) + g[s]*env_vcf)
                        + filter_frequency_max.log2()*(1.0 - g[s])*env_vcf).exp2()
                    }
                    else
                    {
                        (f0.log2()*((1.0 + env_vcf) - g[s]*env_vcf)
                        - FILTER_FREQUENCY_MIN.log2()*(1.0 - g[s])*env_vcf).exp2()
                    }).reduce(|a, b| a + b)
                    .unwrap_or(0.0)
                    + lfo_vcf.iter()
                        .map(|&lfo_vcf| lfo_vcf as f64)
                        .zip(lfo.iter()).map(|(lfo_vcf, &lfo)| lfo_vcf.abs()/a0*if lfo_vcf >= 0.0
                    {
                        (f0.log2()*((1.0 - lfo_vcf) + (0.5*lfo + 0.5)*lfo_vcf)
                        + filter_frequency_max.log2()*(0.5 - 0.5*lfo)*lfo_vcf).exp2()
                    }
                    else
                    {
                        (f0.log2()*((1.0 + lfo_vcf) - (0.5*lfo + 0.5)*lfo_vcf)
                        - FILTER_FREQUENCY_MIN.log2()*(0.5 - 0.5*lfo)*lfo_vcf).exp2()
                    }).reduce(|a, b| a + b)
                    .unwrap_or(0.0)
                };

                for c in 0..CHANNEL_COUNT
                {
                    self.filter[c][i].omega = FILTER_CHANGE*TAU*f + (1.0 - FILTER_CHANGE)*self.filter[c][i].omega;
                }
            }
            for i in poly..POLY_MAX
            {
                for n in 0..ENVELOPE_COUNT
                {
                    self.env[i][n].next(self.rate);
                }
            }
            let g0: [f64; ENVELOPE_COUNT] = ArrayOps::fill(|i| g.iter()
                .map(|g| g[i][s])
                .reduce(|a, b| a.max(b))
                .unwrap_or(1.0));
            for n in 0..LFO_COUNT
            {
                let f0 = self.param.lfo[n].frequency.get() as f64;
                
                let a0 = env_vclfo.iter()
                    .map(|x| x[n].abs() as f64)
                    .reduce(|a, b| a + b)
                    .unwrap_or(0.0)
                    + lfo_vclfo.iter()
                        .map(|x| x[n].abs() as f64)
                        .reduce(|a, b| a + b)
                        .unwrap_or(0.0);
                let f = if a0 == 0.0 || a0 == -0.0
                {
                    f0
                }
                else
                {
                    env_vclfo.iter()
                        .map(|env_vclfo| env_vclfo[n] as f64)
                        .zip(g0.into_iter())
                        .map(|(env_vclfo, g0)| env_vclfo.abs()/a0*if env_vclfo >= 0.0
                    {
                        (f0.log2()*((1.0 - env_vclfo) + g0*env_vclfo)
                        + LFO_FREQUENCY_MAX.log2()*(1.0 - g0)*env_vclfo).exp2()
                    }
                    else
                    {
                        (f0.log2()*((1.0 + env_vclfo) - g0*env_vclfo)
                        - LFO_FREQUENCY_MIN.log2()*(1.0 - g0)*env_vclfo).exp2()
                    }).reduce(|a, b| a + b).unwrap_or(0.0)
                    + 
                    lfo_vclfo.iter()
                        .map(|lfo_vclfo| lfo_vclfo[n] as f64)
                        .zip(lfo.into_iter())
                        .map(|(lfo_vclfo, lfo)| lfo_vclfo.abs()/a0*if lfo_vclfo >= 0.0
                    {
                        (f0.log2()*((1.0 - lfo_vclfo) + (0.5*lfo + 0.5)*lfo_vclfo)
                        + LFO_FREQUENCY_MAX.log2()*(0.5 - 0.5*lfo)*lfo_vclfo).exp2()
                    }
                    else
                    {
                        (f0.log2()*((1.0 + lfo_vclfo) - (0.5*lfo + 0.5)*lfo_vclfo)
                        - LFO_FREQUENCY_MIN.log2()*(0.5 - 0.5*lfo)*lfo_vclfo).exp2()
                    }).reduce(|a, b| a + b).unwrap_or(0.0)
                };
                
                self.lfo[n].omega = TAU*f;
            }
            for i in 0..poly
            {
                for n in 0..ENVELOPE_COUNT
                {
                    g[i][n][s] = 1.0 - env_vca[n] as f64 + env_vca[n] as f64*g[i][n][s];
                }
            }
            if mix_mode == 1
            {
                for i in 0..poly
                {
                    if self.osc[i][0].theta >= TAU*(1.0 - self.osc[i][0].omega()/self.rate)
                    {
                        for n in 1..OSCILLATOR_COUNT
                        {
                            self.osc[i][n].theta = self.osc[i][0].theta;
                        }
                    }
                }
            }
            let y: [f64; CHANNEL_COUNT] = (0..poly).map(|i| {
                        let y: [f64; CHANNEL_COUNT] = (0..OSCILLATOR_COUNT).map(|n| {
                                let g_tot = (0..ENVELOPE_COUNT).map(|m| g[i][m][s]).reduce(|a, b| a*b).unwrap_or(1.0);
                                if g_tot == 0.0 || (mix[n] == 0.0 && mix_mode != 2)
                                {
                                    self.osc[i][n].step(self.rate);
                                    [0.0, 0.0]
                                }
                                else
                                {
                                    let y = g_tot
                                    *if mix_mode != 2
                                    {
                                        mix[n] as f64*self.osc[i][n].next(self.rate)
                                    }
                                    else
                                    {
                                        mix[n] as f64*self.osc[i][n].next(self.rate) + (1.0 - mix[n] as f64)
                                    }
                                    *lfo_vca.iter()
                                        .map(|&lfo_vca| lfo_vca as f64)
                                        .zip(lfo.iter())
                                        .map(|(lfo_vca, lfo)|
                                            1.0 - lfo_vca.abs() + lfo_vca.abs()*(0.5 + lfo_vca.signum()*lfo).max(0.0)
                                        ).reduce(|a, b| a*b)
                                        .unwrap_or(1.0);
                                    
                                    let p = pan[n] as f64;
                                    [(1.0 - p)*y, p*y]
                                }
                            })
                            .reduce(if mix_mode != 2
                                {
                                    |a: [f64; CHANNEL_COUNT], b: [f64; CHANNEL_COUNT]|
                                        ArrayOps::fill(|n| a[n] + b[n])
                                }
                                else
                                {
                                    |a: [f64; CHANNEL_COUNT], b: [f64; CHANNEL_COUNT]|
                                        ArrayOps::fill(|n| a[n] * b[n])
                                }
                            ).unwrap_or([0.0, 0.0]);
                        return ArrayOps::fill(|c| self.filter[c][i].filter(self.rate, y[c])[filter_type])
                    })
                    .reduce(|a: [f64; 2], b| a.add_each(b))
                    .unwrap_or([0.0, 0.0]);
            if mix_mode != 2
            {
                [GAIN*vol as f64*y[0], GAIN*vol as f64*y[1]]
            }
            else
            {
                let mixtot = mix.into_iter().reduce(|a, b| a + b).unwrap_or(0.0) as f64;
                [GAIN*vol as f64*y[0]*mixtot, GAIN*vol as f64*y[1]*mixtot]
            }
        }).collect();

        for (c, output_channel) in outputs.into_iter().enumerate()
        {
            for (output_sample, &y) in output_channel.into_iter().zip(y.iter())
            {
                *output_sample = y[c] as f32
            }
        }

        /*let g0 = g.iter()
            .map(|v| v.clone())
            .reduce(|a, b| (0..samples)
                .map(|s| a[s].max(b[s]))
                .collect::<Vec<f32>>()
            ).unwrap();*/

        /*let events: Vec<[*mut Event; POLY_MAX + 1]> = (0..samples).map(|s|
            ArrayOps::fill(|i| {
                let value = if i == 0
                {
                    g0[s]
                }
                else
                {
                    g[i][s]
                };
                let system_data: *mut u8 = &mut value.to_le_bytes()[0];
                unsafe
                {
                    std::mem::transmute(&mut SysExEvent {
                        event_type: vst::api::EventType::_Trigger,
                        byte_size: 4,
                        data_size: 4,
                        delta_frames: s as i32,
                        _flags: 0,
                        _reserved1: 0,
                        system_data,
                        _reserved2: 0,
                    })
                }
            })
        ).collect();

        for s in 0..samples
        {
            for i in 0..(POLY_MAX + 1)/2
            {
                let events: [*mut Event; 2] = ArrayOps::fill(|j| events[s][i*2 + j] as *mut Event);
                self.host.process_events(&Events {
                    num_events: 2,
                    _reserved: 0,
                    events
                })
            }
        }*/

        /*self.host.begin_edit(0);
        self.host.automate(0, *g0.last().unwrap_or(&0.0));
        self.host.end_edit(0);

        for i in 0..POLY_MAX
        {
            self.host.begin_edit(i as i32 + 1);
            self.host.automate(i as i32 + 1, *g[i].last().unwrap_or(&0.0));
            self.host.end_edit(i as i32 + 1);
        }*/
    }

    fn get_parameter_object(&mut self) -> Arc<dyn PluginParameters>
    {
        self.param.clone()
    }
}

plugin_main!(MekkaSynthPlugin);