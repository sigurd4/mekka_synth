use std::sync::atomic::{AtomicU8, Ordering, AtomicI8, AtomicUsize, AtomicPtr};

use array_math::ArrayOps;
use vst::prelude::PluginParameters;
use vst::util::AtomicFloat;

use super::*;

const FILTER_RESONANCE_CURVE: f32 = 4.0;
const OCTAVE_MAX: i8 = 4;
const OCTAVE_MIN: i8 = -4;

#[derive(Clone, Copy)]
#[repr(i32)]
pub enum Control
{
    OSCSelect,
    LFOSelect,
    EnvelopeSelect,
    VCLFOSelect,

    Key,
    Tuning,
    Polyphony,
    Portamento,
    MixMode,
    Volume,

    OSCWaveform,
    OSCDutyCycle,
    OSCDetune,
    OSCOctave,
    OSCMix,
    OSCPan,
    EnvelopeVCO,
    LFOVCO,
    
    LFOWaveform,
    LFODutyCycle,
    LFOFrequency,

    EnvelopeVCLFO,
    LFOVCLFO,

    EnvelopeAttack,
    EnvelopeDecay,
    EnvelopeSustain,
    EnvelopeRelease,

    FilterType,
    FilterFrequency,
    FilterResonance,
    EnvelopeVCF,
    LFOVCF,

    EnvelopeVCA,
    LFOVCA,
}

impl Control
{
    pub const VARIANT_COUNT: usize = core::mem::variant_count::<Control>();

    pub const CONTROLS: [Self; Self::VARIANT_COUNT] = [
        Self::OSCSelect,
        Self::LFOSelect,
        Self::EnvelopeSelect,
        Self::VCLFOSelect,
        
        Self::OSCWaveform,
        Self::OSCDutyCycle,
        Self::OSCDetune,
        Self::OSCOctave,

        Self::OSCMix,
        Self::OSCPan,
        Self::Volume,

        Self::Key,
        Self::Tuning,
        Self::Polyphony,
        Self::Portamento,
        Self::MixMode,
        
        Self::FilterType,
        Self::FilterFrequency,
        Self::FilterResonance,
        
        Self::EnvelopeAttack,
        Self::EnvelopeDecay,
        Self::EnvelopeSustain,

        Self::EnvelopeRelease,
        Self::EnvelopeVCA,
        Self::EnvelopeVCO,

        Self::EnvelopeVCF,
        Self::EnvelopeVCLFO,
        
        Self::LFOWaveform,
        Self::LFODutyCycle,
        Self::LFOFrequency,
        Self::LFOVCA,
        
        Self::LFOVCO,
        Self::LFOVCF,
        Self::LFOVCLFO,
    ];

    fn from(i: i32) -> Self
    {
        Self::CONTROLS[i as usize]
    }
}

#[derive(Debug)]
pub struct SelectionParameters
{
    pub osc: AtomicU8,
    pub lfo: AtomicU8,
    pub envelope: AtomicU8,
    pub vclfo: AtomicU8,
}

#[derive(Debug)]
pub struct OscillatorParameters
{
    pub waveform: AtomicU8,
    pub duty_cycle: AtomicFloat,
    pub detune: AtomicFloat,
    pub octave: AtomicI8,
    pub mix: AtomicFloat,
    pub pan: AtomicFloat,
}

impl OscillatorParameters
{
    pub fn waveform(&self) -> Waveform
    {
        Waveform::WAVEFORMS[self.waveform.load(Ordering::Relaxed) as usize]
    }

    pub fn get_preset_data(&self) -> Vec<u8>
    {
        let mut data = vec![];

        data.append(&mut self.waveform.load(Ordering::Relaxed).to_le_bytes().to_vec());
        data.append(&mut self.duty_cycle.get().to_le_bytes().to_vec());
        data.append(&mut self.detune.get().to_le_bytes().to_vec());
        data.append(&mut self.octave.load(Ordering::Relaxed).to_le_bytes().to_vec());
        data.append(&mut self.mix.get().to_le_bytes().to_vec());
        data.append(&mut self.pan.get().to_le_bytes().to_vec());

        data
    }
    
    pub fn load_preset_data<I: Iterator<Item = u8>>(&self, mut data: &mut I)
    {
        self.waveform.store(u8::from_le_bytes([(); _].map(|()| data.next().unwrap())), Ordering::Relaxed);
        self.duty_cycle.set(f32::from_le_bytes([(); _].map(|()| data.next().unwrap())));
        self.detune.set(f32::from_le_bytes([(); _].map(|()| data.next().unwrap())));
        self.octave.store(i8::from_le_bytes([(); _].map(|()| data.next().unwrap())), Ordering::Relaxed);
        self.mix.set(f32::from_le_bytes([(); _].map(|()| data.next().unwrap())));
        self.pan.set(f32::from_le_bytes([(); _].map(|()| data.next().unwrap())));
    }
}

#[derive(Debug)]
pub struct EnvelopeParameters
{
    pub attack: AtomicFloat,
    pub decay: AtomicFloat,
    pub sustain: AtomicFloat,
    pub release: AtomicFloat,

    pub routing: RoutingParameters
}

impl EnvelopeParameters
{
    pub fn get_preset_data(&self) -> Vec<u8>
    {
        let mut data = vec![];

        data.append(&mut self.attack.get().to_le_bytes().to_vec());
        data.append(&mut self.decay.get().to_le_bytes().to_vec());
        data.append(&mut self.sustain.get().to_le_bytes().to_vec());
        data.append(&mut self.release.get().to_le_bytes().to_vec());
        
        data.append(&mut self.routing.get_preset_data());

        data
    }

    pub fn load_preset_data<I: Iterator<Item = u8>>(&self, mut data: &mut I)
    {
        self.attack.set(f32::from_le_bytes([(); _].map(|()| data.next().unwrap())));
        self.decay.set(f32::from_le_bytes([(); _].map(|()| data.next().unwrap())));
        self.sustain.set(f32::from_le_bytes([(); _].map(|()| data.next().unwrap())));
        self.release.set(f32::from_le_bytes([(); _].map(|()| data.next().unwrap())));

        self.routing.load_preset_data(data);
    }
}

#[derive(Debug)]
pub struct LFOParameters
{
    pub waveform: AtomicU8,
    pub duty_cycle: AtomicFloat,
    pub frequency: AtomicFloat,

    pub routing: RoutingParameters
}

impl LFOParameters
{
    pub fn waveform(&self) -> Waveform
    {
        Waveform::WAVEFORMS[self.waveform.load(Ordering::Relaxed) as usize]
    }
    
    pub fn get_preset_data(&self) -> Vec<u8>
    {
        let mut data = vec![];

        data.append(&mut self.waveform.load(Ordering::Relaxed).to_le_bytes().to_vec());
        data.append(&mut self.duty_cycle.get().to_le_bytes().to_vec());
        data.append(&mut self.frequency.get().to_le_bytes().to_vec());
        
        data.append(&mut self.routing.get_preset_data());

        data
    }
    
    pub fn load_preset_data<I: Iterator<Item = u8>>(&self, mut data: &mut I)
    {
        self.waveform.store(u8::from_le_bytes([(); _].map(|()| data.next().unwrap())), Ordering::Relaxed);
        self.duty_cycle.set(f32::from_le_bytes([(); _].map(|()| data.next().unwrap())));
        self.frequency.set(f32::from_le_bytes([(); _].map(|()| data.next().unwrap())));

        self.routing.load_preset_data(data);
    }
}

#[derive(Debug)]
pub struct RoutingParameters
{
    pub vca: AtomicFloat,
    pub vcf: AtomicFloat,
    pub vco: AtomicFloat,
    pub vclfo: [AtomicFloat; LFO_COUNT],
}

impl RoutingParameters
{
    pub fn get_preset_data(&self) -> Vec<u8>
    {
        let mut data = vec![];

        data.append(&mut self.vca.get().to_le_bytes().to_vec());
        data.append(&mut self.vcf.get().to_le_bytes().to_vec());
        data.append(&mut self.vco.get().to_le_bytes().to_vec());

        for vclfo in self.vclfo.iter()
        {
            data.append(&mut vclfo.get().to_le_bytes().to_vec());
        }

        data
    }
    
    pub fn load_preset_data<I: Iterator<Item = u8>>(&self, mut data: &mut I)
    {
        self.vco.set(f32::from_le_bytes([(); _].map(|()| data.next().unwrap())));
        self.vcf.set(f32::from_le_bytes([(); _].map(|()| data.next().unwrap())));
        self.vco.set(f32::from_le_bytes([(); _].map(|()| data.next().unwrap())));

        for vclfo in self.vclfo.iter()
        {
            vclfo.set(f32::from_le_bytes([(); _].map(|()| data.next().unwrap())));
        }
    }
}

#[derive(Debug)]
pub struct FilterParameters
{
    pub type_: AtomicU8,
    pub frequency: AtomicFloat,
    pub resonance: AtomicFloat,
}

impl FilterParameters
{
    pub fn get_preset_data(&self) -> Vec<u8>
    {
        let mut data = vec![];

        data.push(self.type_.load(Ordering::Relaxed));
        data.append(&mut self.frequency.get().to_le_bytes().to_vec());
        data.append(&mut self.resonance.get().to_le_bytes().to_vec());

        data
    }
    
    pub fn load_preset_data<I: Iterator<Item = u8>>(&self, mut data: &mut I)
    {
        self.type_.store(u8::from_le_bytes([(); _].map(|()| data.next().unwrap())), Ordering::Relaxed);
        self.frequency.set(f32::from_le_bytes([(); _].map(|()| data.next().unwrap())));
        self.resonance.set(f32::from_le_bytes([(); _].map(|()| data.next().unwrap())));
    }
}

#[derive(Debug)]
pub struct MekkaSynthParameters
{
    pub select: SelectionParameters,

    pub key: AtomicU8,
    pub tuning: AtomicUsize,
    pub polyphony: AtomicU8,
    pub portamento: AtomicFloat,
    pub mix_mode: AtomicU8,
    pub volume: AtomicFloat,

    pub osc: [OscillatorParameters; OSCILLATOR_COUNT],
    pub envelope: [EnvelopeParameters; ENVELOPE_COUNT],
    pub lfo: [LFOParameters; LFO_COUNT],

    pub filter: FilterParameters,

    pub filter_frequency_max: AtomicFloat,
    pub scales: Arc<BTreeMap<String, RwLock<Option<Scale>>>>
}

impl MekkaSynthParameters
{
    pub fn tuning_name(&self) -> Option<&String>
    {
        self.scales.keys()
            .nth(self.tuning.load(Ordering::Relaxed))
    }

    pub fn tuning(&self) -> Option<Scale>
    {
        self.tuning_name().and_then(|name| MekkaSynthPlugin::fetch_scale(&self.scales, name).unwrap())
    }

    fn current_osc(&self) -> &OscillatorParameters
    {
        &self.osc[self.select.osc.load(Ordering::Relaxed) as usize]
    }

    fn current_env(&self) -> &EnvelopeParameters
    {
        &self.envelope[self.select.envelope.load(Ordering::Relaxed) as usize]
    }

    fn current_lfo(&self) -> &LFOParameters
    {
        &self.lfo[self.select.lfo.load(Ordering::Relaxed) as usize]
    }

    fn load_preset_data_checked(&self, data: &[u8])
    {
        let mut data = data.into_iter()
            .map(|&d| d);
        
        self.polyphony.store(u8::from_le_bytes([(); _].map(|()| data.next().unwrap())), Ordering::Relaxed);
        self.portamento.set(f32::from_le_bytes([(); _].map(|()| data.next().unwrap())));
        self.mix_mode.store(u8::from_le_bytes([(); _].map(|()| data.next().unwrap())), Ordering::Relaxed);
        self.volume.set(f32::from_le_bytes([(); _].map(|()| data.next().unwrap())));
        
        for osc in self.osc.iter()
        {
            osc.load_preset_data(&mut data);
        }

        for env in self.envelope.iter()
        {
            env.load_preset_data(&mut data);
        }
        
        for lfo in self.lfo.iter()
        {
            lfo.load_preset_data(&mut data);
        }
        
        self.filter.load_preset_data(&mut data);
        
        self.key.store(u8::from_le_bytes([(); _].map(|()| data.next().unwrap())), Ordering::Relaxed);

        let len = usize::from_le_bytes([(); _].map(|()| data.next().unwrap()));
        let tuning_name = String::from_utf8((0..len).map(|_| u8::from_le_bytes([(); _].map(|()| data.next().unwrap()))).collect()).unwrap();
        self.tuning.store(
            self.scales.keys()
                .enumerate()
                .filter(|(_, scale_name)| **scale_name == tuning_name)
                .map(|(i, _)| i)
                .next()
                .unwrap_or(0),
            Ordering::Relaxed
        )
    }
}

impl PluginParameters for MekkaSynthParameters
{
    fn get_parameter_label(&self, index: i32) -> String
    {
        match Control::from(index)
        {
            Control::OSCSelect => "".to_string(),
            Control::LFOSelect => "".to_string(),
            Control::EnvelopeSelect => "".to_string(),
            Control::VCLFOSelect => "".to_string(),

            Control::Key => "".to_string(),
            Control::Tuning => self.tuning().map(|tuning| tuning.name.clone()).unwrap_or_default(),
            Control::Polyphony => "tones".to_string(),
            Control::Portamento => "s".to_string(),
            Control::MixMode => "".to_string(),
            Control::Volume => "%".to_string(),
        
            Control::OSCWaveform => "".to_string(),
            Control::OSCDutyCycle => "%".to_string(),
            Control::OSCDetune => "cents".to_string(),
            Control::OSCOctave => "octaves".to_string(),
            Control::OSCMix => "%".to_string(),
            Control::OSCPan => "%".to_string(),
            Control::EnvelopeVCO => "cents".to_string(),
            Control::LFOVCO => "cents".to_string(),
            
            Control::LFOWaveform => "".to_string(),
            Control::LFODutyCycle => "%".to_string(),
            Control::LFOFrequency => "Hz".to_string(),
        
            Control::EnvelopeVCLFO => "Hz".to_string(),
            Control::LFOVCLFO => "Hz".to_string(),
        
            Control::EnvelopeAttack => "s".to_string(),
            Control::EnvelopeDecay => "s".to_string(),
            Control::EnvelopeSustain => "%".to_string(),
            Control::EnvelopeRelease => "s".to_string(),
        
            Control::FilterType => "".to_string(),
            Control::FilterFrequency => "Hz".to_string(),
            Control::FilterResonance => "".to_string(),
            Control::EnvelopeVCF => "Hz".to_string(),
            Control::LFOVCF => "Hz".to_string(),
        
            Control::EnvelopeVCA => "%".to_string(),
            Control::LFOVCA => "%".to_string()
        }
    }

    fn get_parameter_text(&self, index: i32) -> String
    {
        match Control::from(index)
        {
            Control::OSCSelect => format!("OSC {}", self.select.osc.load(Ordering::Relaxed) + 1),
            Control::LFOSelect => format!("LFO {}", self.select.lfo.load(Ordering::Relaxed) + 1),
            Control::EnvelopeSelect => format!("Envelope {}", self.select.envelope.load(Ordering::Relaxed) + 1),
            Control::VCLFOSelect => format!("VCLFO {}", self.select.vclfo.load(Ordering::Relaxed) + 1),

            Control::Key => KEYS[self.key.load(Ordering::Relaxed) as usize].to_string(),
            Control::Tuning => self.tuning_name().map(|name| name.clone()).unwrap_or("12edo".to_string()),
            Control::Polyphony => format!("{}", self.polyphony.load(Ordering::Relaxed)),
            Control::Portamento => format!("{:.3}", self.portamento.get()),
            Control::MixMode => match self.mix_mode.load(Ordering::Relaxed)
            {
                0 => "Mix".to_string(),
                1 => "Sync".to_string(),
                2 => "Ring-Mod".to_string(),
                _ => "ERR".to_string()
            },
            Control::Volume => format!("{:.3}", 100.0*self.volume.get()),
        
            Control::OSCWaveform => match self.current_osc().waveform()
            {
                Waveform::Sine => "Sine".to_string(),
                Waveform::Triangle => "Triangle".to_string(),
                Waveform::Sawtooth => "Sawtooth".to_string(),
                Waveform::Square => "Square".to_string(),
                Waveform::Noise => "Noise".to_string(),
            },
            Control::OSCDutyCycle => format!("{:.3}", 100.0*self.current_osc().duty_cycle.get()),
            Control::OSCDetune => format!("{:.3}", 100.0*self.current_osc().detune.get()),
            Control::OSCOctave => {
                let i = self.select.osc.load(Ordering::Relaxed) as usize;
                let o = self.current_osc().octave.load(Ordering::Relaxed);
                format!("{}{}", if o > 0 {"+"} else {""}, o)
            },
            Control::OSCMix => format!("{:.3}", 100.0*self.current_osc().mix.get()),
            Control::OSCPan => {
                let i = self.select.osc.load(Ordering::Relaxed) as usize;
                let p = 2.0*self.current_osc().pan.get() - 1.0;
                if p == 0.0 || p == -0.0
                {
                    "Center".to_string()
                }
                else
                {
                    format!("{:.3} {}", 100.0*p.abs(), if p > 0.0 {"Right"} else {"Left"})
                }
            },
            Control::EnvelopeVCO => format!("{:.3}", 100.0*self.current_env().routing.vco.get()),
            Control::LFOVCO => format!("{:.3}", 100.0*self.current_lfo().routing.vco.get()),
            
            Control::LFOWaveform => match self.current_lfo().waveform()
            {
                Waveform::Sine => "Sine".to_string(),
                Waveform::Triangle => "Triangle".to_string(),
                Waveform::Sawtooth => "Sawtooth".to_string(),
                Waveform::Square => "Square".to_string(),
                Waveform::Noise => "Noise".to_string(),
            },
            Control::LFODutyCycle => format!("{:.3}", 100.0*self.current_lfo().duty_cycle.get()),
            Control::LFOFrequency => format!("{:.3}", self.current_lfo().frequency.get()),
        
            Control::EnvelopeVCLFO => {
                let vclfo_select = self.select.vclfo.load(Ordering::Relaxed) as usize;
                let env_select = self.select.envelope.load(Ordering::Relaxed) as usize;
                let f0 = self.lfo[vclfo_select].frequency.get();
                let env_vclfo = self.envelope.each_ref()
                    .map(|env| env.routing.vclfo[vclfo_select].get());
                let lfo_vclfo = self.lfo.each_ref()
                    .map(|lfo| lfo.routing.vclfo[vclfo_select].get());
                
                let a = env_vclfo[env_select].abs()/(EPSILON as f32
                        + env_vclfo.into_iter().map(|a| a.abs()).reduce(|a, b| a + b).unwrap_or(0.0)
                        + lfo_vclfo.into_iter().map(|a| a.abs()).reduce(|a, b| a + b).unwrap_or(0.0)
                    );
                let b = 1.0 - a;

                let f = if env_vclfo[env_select] >= 0.0
                {
                    (f0.log2()*(1.0 - env_vclfo[env_select])
                    + (LFO_FREQUENCY_MAX as f32).log2()*env_vclfo[env_select]).exp2()
                }
                else
                {
                    (f0.log2()*(1.0 + env_vclfo[env_select])
                    - (LFO_FREQUENCY_MIN as f32).log2()*env_vclfo[env_select]).exp2()
                }*a
                + f0*b;
                format!("{:.3}", f)
            },
            Control::LFOVCLFO => {
                let vclfo_select = self.select.vclfo.load(Ordering::Relaxed) as usize;
                let lfo_select = self.select.lfo.load(Ordering::Relaxed) as usize;
                let f0 = self.lfo[vclfo_select].frequency.get();
                let env_vclfo = self.envelope.each_ref()
                    .map(|env| env.routing.vclfo[vclfo_select].get());
                let lfo_vclfo = self.lfo.each_ref()
                    .map(|lfo| lfo.routing.vclfo[vclfo_select].get());
                
                let a = lfo_vclfo[lfo_select].abs()/(EPSILON as f32
                        + env_vclfo.into_iter().map(|a| a.abs()).reduce(|a, b| a + b).unwrap_or(0.0)
                        + lfo_vclfo.into_iter().map(|a| a.abs()).reduce(|a, b| a + b).unwrap_or(0.0)
                    );
                let b = 1.0 - a;

                let f = if lfo_vclfo[lfo_select] >= 0.0
                {
                    (f0.log2()*(1.0 - lfo_vclfo[lfo_select])
                    + (LFO_FREQUENCY_MAX as f32).log2()*lfo_vclfo[lfo_select]).exp2()
                }
                else
                {
                    (f0.log2()*(1.0 + lfo_vclfo[lfo_select])
                    - (LFO_FREQUENCY_MIN as f32).log2()*lfo_vclfo[lfo_select]).exp2()
                }*a
                + f0*b;
                format!("{:.3}", f)
            },
        
            Control::EnvelopeAttack => format!("{:.3}", self.current_env().attack.get()),
            Control::EnvelopeDecay => format!("{:.3}", self.current_env().decay.get()),
            Control::EnvelopeSustain => format!("{:.3}", 100.0*self.current_env().sustain.get()),
            Control::EnvelopeRelease => format!("{:.3}", self.current_env().release.get()),
        
            Control::FilterType => match self.filter.type_.load(Ordering::Relaxed)
            {
                0 => "Low-pass".to_string(),
                1 => "Notch".to_string(),
                2 => "High-pass".to_string(),
                _ => "ERR".to_string()
            },
            Control::FilterFrequency => format!("{:.3}", self.filter.frequency.get()),
            Control::FilterResonance => format!("{:.3}", self.filter.resonance.get()),
            Control::EnvelopeVCF => {
                let env_select = self.select.envelope.load(Ordering::Relaxed) as usize;
                let f0 = self.filter.frequency.get();
                let env_vcf = self.envelope.each_ref()
                    .map(|env| env.routing.vcf.get());
                let lfo_vcf = self.lfo.each_ref()
                    .map(|lfo| lfo.routing.vcf.get());
                
                let a = env_vcf[env_select].abs()/(EPSILON as f32
                        + env_vcf.into_iter().map(|a| a.abs()).reduce(|a, b| a + b).unwrap_or(0.0)
                        + lfo_vcf.into_iter().map(|a| a.abs()).reduce(|a, b| a + b).unwrap_or(0.0)
                    );
                let b = 1.0 - a;

                let f = if env_vcf[env_select] >= 0.0
                {
                    (f0.log2()*(1.0 - env_vcf[env_select])
                    + self.filter_frequency_max.get().log2()*env_vcf[env_select]).exp2()
                }
                else
                {
                    (f0.log2()*(1.0 + env_vcf[env_select])
                    - (FILTER_FREQUENCY_MIN as f32).log2()*env_vcf[env_select]).exp2()
                }*a
                + f0*b;
                format!("{:.3}", f)
            },
            Control::LFOVCF => {
                let lfo_select = self.select.lfo.load(Ordering::Relaxed) as usize;
                let f0 = self.filter.frequency.get();
                let env_vcf = self.envelope.each_ref()
                    .map(|env| env.routing.vcf.get());
                let lfo_vcf = self.lfo.each_ref()
                    .map(|lfo| lfo.routing.vcf.get());
                
                let a = lfo_vcf[lfo_select].abs()/(EPSILON as f32
                        + env_vcf.into_iter().map(|a| a.abs()).reduce(|a, b| a + b).unwrap_or(0.0)
                        + lfo_vcf.into_iter().map(|a| a.abs()).reduce(|a, b| a + b).unwrap_or(0.0)
                    );
                let b = 1.0 - a;

                let f = if lfo_vcf[lfo_select] >= 0.0
                {
                    (f0.log2()*(1.0 - lfo_vcf[lfo_select])
                    + self.filter_frequency_max.get().log2()*lfo_vcf[lfo_select]).exp2()
                }
                else
                {
                    (f0.log2()*(1.0 + lfo_vcf[lfo_select])
                    - (FILTER_FREQUENCY_MIN as f32).log2()*lfo_vcf[lfo_select]).exp2()
                }*a
                + f0*b;
                format!("{:.3}", f)
            },
        
            Control::EnvelopeVCA => format!("{:.3}", 100.0*self.current_env().routing.vca.get()),
            Control::LFOVCA => format!("{:.3}", 100.0*self.current_lfo().routing.vca.get()),
        }
    }

    fn get_parameter_name(&self, index: i32) -> String
    {
        match Control::from(index)
        {
            Control::OSCSelect => "OSC Select".to_string(),
            Control::LFOSelect => "LFO Select".to_string(),
            Control::EnvelopeSelect => "Envelope Select".to_string(),
            Control::VCLFOSelect => "VCLFO Select".to_string(),

            Control::Key => "Key".to_string(),
            Control::Tuning => "Tuning".to_string(),
            Control::Polyphony => "Polyphony".to_string(),
            Control::Portamento => "Portamento".to_string(),
            Control::MixMode => "Mix Mode".to_string(),
            Control::Volume => "Volume".to_string(),
        
            Control::OSCWaveform => format!("OSC {} Waveform", self.select.osc.load(Ordering::Relaxed) + 1),
            Control::OSCDutyCycle => format!("OSC {} Duty Cycle", self.select.osc.load(Ordering::Relaxed) + 1),
            Control::OSCDetune => format!("OSC {} Detune", self.select.osc.load(Ordering::Relaxed) + 1),
            Control::OSCOctave => format!("OSC {} Octave", self.select.osc.load(Ordering::Relaxed) + 1),
            Control::OSCMix => format!("OSC {} Mix", self.select.osc.load(Ordering::Relaxed) + 1),
            Control::OSCPan => format!("OSC {} Pan", self.select.osc.load(Ordering::Relaxed) + 1),
            Control::EnvelopeVCO => format!("Envelope {} -> VCO", self.select.envelope.load(Ordering::Relaxed) + 1),
            Control::LFOVCO => format!("LFO {} -> VCO", self.select.lfo.load(Ordering::Relaxed) + 1),
            
            Control::LFOWaveform => format!("LFO {} Waveform", self.select.lfo.load(Ordering::Relaxed) + 1),
            Control::LFODutyCycle => format!("LFO {} Duty Cycle", self.select.lfo.load(Ordering::Relaxed) + 1),
            Control::LFOFrequency => format!("LFO {} Frequency", self.select.lfo.load(Ordering::Relaxed) + 1),
        
            Control::EnvelopeVCLFO => format!("Envelope {} -> VCLFO {}", self.select.envelope.load(Ordering::Relaxed) + 1, self.select.vclfo.load(Ordering::Relaxed) + 1),
            Control::LFOVCLFO => format!("LFO {} -> VCLFO {}", self.select.lfo.load(Ordering::Relaxed) + 1, self.select.vclfo.load(Ordering::Relaxed) + 1),
        
            Control::EnvelopeAttack => format!("Envelope {} Attack", self.select.envelope.load(Ordering::Relaxed) + 1),
            Control::EnvelopeDecay => format!("Envelope {} Decay", self.select.envelope.load(Ordering::Relaxed) + 1),
            Control::EnvelopeSustain => format!("Envelope {} Sustain", self.select.envelope.load(Ordering::Relaxed) + 1),
            Control::EnvelopeRelease => format!("Envelope {} Release", self.select.envelope.load(Ordering::Relaxed) + 1),
        
            Control::FilterType => "Filter Type".to_string(),
            Control::FilterFrequency => "Filter Frequency".to_string(),
            Control::FilterResonance => "Filter Resonance".to_string(),
            Control::EnvelopeVCF => format!("Envelope {} -> VCF", self.select.envelope.load(Ordering::Relaxed) + 1),
            Control::LFOVCF => format!("LFO {} -> VCF", self.select.lfo.load(Ordering::Relaxed) + 1),
        
            Control::EnvelopeVCA => format!("Envelope {} -> VCA", self.select.envelope.load(Ordering::Relaxed) + 1),
            Control::LFOVCA => format!("LFO {} -> VCA", self.select.lfo.load(Ordering::Relaxed) + 1),
        }
    }

    /// Get the value of parameter at `index`. Should be value between 0.0 and 1.0.
    fn get_parameter(&self, index: i32) -> f32
    {
        match Control::from(index)
        {
            Control::OSCSelect => self.select.osc.load(Ordering::Relaxed) as f32/(OSCILLATOR_COUNT - 1) as f32,
            Control::LFOSelect => self.select.lfo.load(Ordering::Relaxed) as f32/(LFO_COUNT - 1) as f32,
            Control::EnvelopeSelect => self.select.envelope.load(Ordering::Relaxed) as f32/(LFO_COUNT - 1) as f32,
            Control::VCLFOSelect => self.select.vclfo.load(Ordering::Relaxed) as f32/(LFO_COUNT - 1) as f32,

            Control::Key => self.key.load(Ordering::Relaxed) as f32/11.0,
            Control::Tuning => self.tuning.load(Ordering::Relaxed) as f32/(self.scales.len().saturating_sub(2) + 1) as f32,
            Control::Polyphony => ((self.polyphony.load(Ordering::Relaxed) - POLY_MIN as u8) as f32)/((POLY_MAX - POLY_MIN) as f32),
            Control::Portamento => (self.portamento.get().log2() - (PORTAMENTO_MIN as f32).log2())/((PORTAMENTO_MAX as f32).log2() - (PORTAMENTO_MIN as f32).log2()),
            Control::MixMode => (self.mix_mode.load(Ordering::Relaxed) as f32)/2.0,
            Control::Volume => self.volume.get(),
        
            Control::OSCWaveform => (self.current_osc().waveform.load(Ordering::Relaxed) as f32)/((Waveform::WAVEFORMS.len() - 1) as f32),
            Control::OSCDutyCycle => self.current_osc().duty_cycle.get(),
            Control::OSCDetune => (self.current_osc().detune.get() - DETUNE_MIN as f32)/(DETUNE_MAX as f32 - DETUNE_MIN as f32),
            Control::OSCOctave => (self.current_osc().octave.load(Ordering::Relaxed) - OCTAVE_MIN) as f32/(OCTAVE_MAX - OCTAVE_MIN) as f32,
            Control::OSCMix => self.current_osc().mix.get(),
            Control::OSCPan => self.current_osc().pan.get(),
            Control::EnvelopeVCO => (self.current_env().routing.vco.get() - ENVELOPE_VCO_MIN as f32)/(ENVELOPE_VCO_MAX as f32 - ENVELOPE_VCO_MIN as f32),
            Control::LFOVCO => (self.current_lfo().routing.vco.get() - ENVELOPE_VCO_MIN as f32)/(ENVELOPE_VCO_MAX as f32 - ENVELOPE_VCO_MIN as f32),
            
            Control::LFOWaveform => (self.current_lfo().waveform.load(Ordering::Relaxed) as f32)/((Waveform::WAVEFORMS.len() - 1) as f32),
            Control::LFODutyCycle => self.current_lfo().duty_cycle.get(),
            Control::LFOFrequency => (self.current_lfo().frequency.get().log2() - (LFO_FREQUENCY_MIN as f32).log2())/((LFO_FREQUENCY_MAX as f32).log2() - (LFO_FREQUENCY_MIN as f32).log2()),
        
            Control::EnvelopeVCLFO => (self.current_env().routing.vclfo[self.select.vclfo.load(Ordering::Relaxed) as usize].get() + 1.0)/2.0,
            Control::LFOVCLFO => (self.current_lfo().routing.vclfo[self.select.vclfo.load(Ordering::Relaxed) as usize].get() + 1.0)/2.0,
        
            Control::EnvelopeAttack => (self.current_env().attack.get().log2() - (ATTACK_MIN as f32).log2())/((ATTACK_MAX as f32).log2() - (ATTACK_MIN as f32).log2()),
            Control::EnvelopeDecay => (self.current_env().decay.get().log2() - (DECAY_MIN as f32).log2())/((DECAY_MAX as f32).log2() - (DECAY_MIN as f32).log2()),
            Control::EnvelopeSustain => self.current_env().sustain.get(),
            Control::EnvelopeRelease => (self.current_env().release.get().log2() - (RELEASE_MIN as f32).log2())/((RELEASE_MAX as f32).log2() - (RELEASE_MIN as f32).log2()),
        
            Control::FilterType => (self.filter.type_.load(Ordering::Relaxed) as f32)/2.0,
            Control::FilterFrequency => (self.filter.frequency.get().log2() - (FILTER_FREQUENCY_MIN as f32).log2())/(self.filter_frequency_max.get().log2() - (FILTER_FREQUENCY_MIN as f32).log2()),
            Control::FilterResonance => ((self.filter.resonance.get() - FILTER_RESONANCE_MIN as f32)/(FILTER_RESONANCE_MAX as f32 - FILTER_RESONANCE_MIN as f32)).powf(1.0/FILTER_RESONANCE_CURVE),
            Control::EnvelopeVCF => (self.current_env().routing.vcf.get() + 1.0)/2.0,
            Control::LFOVCF => (self.current_lfo().routing.vcf.get() + 1.0)/2.0,
        
            Control::EnvelopeVCA => self.current_env().routing.vca.get(),
            Control::LFOVCA => (self.current_lfo().routing.vca.get() + 1.0)/2.0,
        }
    }
    
    fn set_parameter(&self, index: i32, value: f32)
    {
        match Control::from(index)
        {
            Control::OSCSelect => self.select.osc.store((value*(OSCILLATOR_COUNT - 1) as f32).round() as u8, Ordering::Relaxed),
            Control::LFOSelect => self.select.lfo.store((value*(LFO_COUNT - 1) as f32).round() as u8, Ordering::Relaxed),
            Control::EnvelopeSelect => self.select.envelope.store((value*(LFO_COUNT - 1) as f32).round() as u8, Ordering::Relaxed),
            Control::VCLFOSelect => self.select.vclfo.store((value*(LFO_COUNT - 1) as f32).round() as u8, Ordering::Relaxed),

            Control::Key => self.key.store((value*12.0).floor() as u8 % 12, Ordering::Relaxed),
            Control::Tuning => self.tuning.store((value*self.scales.len().saturating_sub(1) as f32).round() as usize, Ordering::Relaxed),
            Control::Polyphony => self.polyphony.store((value*((POLY_MAX - POLY_MIN) as f32)).round() as u8 + POLY_MIN as u8, Ordering::Relaxed),
            Control::Portamento => self.portamento.set((value*((PORTAMENTO_MAX as f32).log2() - (PORTAMENTO_MIN as f32).log2()) + (PORTAMENTO_MIN as f32).log2()).exp2()),
            Control::MixMode => self.mix_mode.store((value*2.0).round() as u8, Ordering::Relaxed),
            Control::Volume => self.volume.set(value),
        
            Control::OSCWaveform => self.current_osc().waveform.store((value*(Waveform::WAVEFORMS.len() - 1) as f32).round() as u8, Ordering::Relaxed),
            Control::OSCDutyCycle => self.current_osc().duty_cycle.set(value),
            Control::OSCDetune => self.current_osc().detune.set(value*(DETUNE_MAX - DETUNE_MIN) as f32 + DETUNE_MIN as f32),
            Control::OSCOctave => self.current_osc().octave.store(((value*(OCTAVE_MAX - OCTAVE_MIN) as f32).round() as i8 + OCTAVE_MIN) as i8, Ordering::Relaxed),
            Control::OSCMix => self.current_osc().mix.set(value),
            Control::OSCPan => self.current_osc().pan.set(value),
            Control::EnvelopeVCO => self.current_env().routing.vco.set(value*(ENVELOPE_VCO_MAX - ENVELOPE_VCO_MIN) as f32 + ENVELOPE_VCO_MIN as f32),
            Control::LFOVCO => self.current_lfo().routing.vco.set(value*(ENVELOPE_VCO_MAX - ENVELOPE_VCO_MIN) as f32 + ENVELOPE_VCO_MIN as f32),
            
            Control::LFOWaveform => self.current_lfo().waveform.store((value*((Waveform::WAVEFORMS.len() - 1) as f32)).round() as u8, Ordering::Relaxed),
            Control::LFODutyCycle => self.current_lfo().duty_cycle.set(value),
            Control::LFOFrequency => self.current_lfo().frequency.set((value*(LFO_FREQUENCY_MAX.log2() - LFO_FREQUENCY_MIN.log2()) as f32 + LFO_FREQUENCY_MIN.log2() as f32).exp2()),
        
            Control::EnvelopeVCLFO => self.current_env().routing.vclfo[self.select.vclfo.load(Ordering::Relaxed) as usize].set(value*2.0 - 1.0),
            Control::LFOVCLFO => self.current_lfo().routing.vclfo[self.select.vclfo.load(Ordering::Relaxed) as usize].set(value*2.0 - 1.0),
        
            Control::EnvelopeAttack => self.current_env().attack.set((value*(ATTACK_MAX.log2() - ATTACK_MIN.log2()) as f32 + ATTACK_MIN.log2() as f32).exp2()),
            Control::EnvelopeDecay => self.current_env().decay.set((value*(DECAY_MAX.log2() - DECAY_MIN.log2()) as f32 + DECAY_MIN.log2() as f32).exp2()),
            Control::EnvelopeSustain => self.current_env().sustain.set(value),
            Control::EnvelopeRelease => self.current_env().release.set((value*(RELEASE_MAX.log2() - RELEASE_MIN.log2()) as f32 + RELEASE_MIN.log2() as f32).exp2()),
        
            Control::FilterType => self.filter.type_.store((value*2.0).round() as u8, Ordering::Relaxed),
            Control::FilterFrequency => self.filter.frequency.set((value*(self.filter_frequency_max.get().log2() - FILTER_FREQUENCY_MIN.log2() as f32) + FILTER_FREQUENCY_MIN.log2() as f32).exp2()),
            Control::FilterResonance => self.filter.resonance.set(FILTER_RESONANCE_MIN as f32 + (FILTER_RESONANCE_MAX - FILTER_RESONANCE_MIN) as f32*value.powf(FILTER_RESONANCE_CURVE)),
            Control::EnvelopeVCF => self.current_env().routing.vcf.set(value*2.0 - 1.0),
            Control::LFOVCF => self.current_lfo().routing.vcf.set(value*2.0 - 1.0),
        
            Control::EnvelopeVCA => self.current_env().routing.vca.set(value),
            Control::LFOVCA => self.current_lfo().routing.vca.set(value*2.0 - 1.0),
        }
    }

    fn change_preset(&self, _preset: i32) {}

    fn get_preset_num(&self) -> i32 {
        0
    }

    fn set_preset_name(&self, _name: String) {}

    fn get_preset_name(&self, _preset: i32) -> String
    {
        "".to_string()
    }

    fn can_be_automated(&self, index: i32) -> bool
    {
        index < Control::CONTROLS.len() as i32
    }

    fn get_preset_data(&self) -> Vec<u8>
    {
        let mut data = vec![];

        data.append(&mut self.polyphony.load(Ordering::Relaxed).to_le_bytes().to_vec());
        data.append(&mut self.portamento.get().to_le_bytes().to_vec());
        data.append(&mut self.mix_mode.load(Ordering::Relaxed).to_le_bytes().to_vec());
        data.append(&mut self.volume.get().to_le_bytes().to_vec());
        
        for osc in self.osc.iter()
        {
            data.append(&mut osc.get_preset_data());
        }

        for env in self.envelope.iter()
        {
            data.append(&mut env.get_preset_data());
        }
        
        for lfo in self.lfo.iter()
        {
            data.append(&mut lfo.get_preset_data());
        }
        
        data.append(&mut self.filter.get_preset_data());

        data.append(&mut self.key.load(Ordering::Relaxed).to_le_bytes().to_vec());

        let tuning_name = self.tuning_name();
        data.append(&mut tuning_name.map(|n| n.bytes().len()).unwrap_or(0).to_le_bytes().to_vec());
        data.append(&mut tuning_name.map(|n| n.bytes().collect()).unwrap_or_default());
        
        data
    }

    fn get_bank_data(&self) -> Vec<u8>
    {
        self.get_preset_data()
    }

    fn load_preset_data(&self, data: &[u8])
    {
        self.load_preset_data_checked(data);
    }

    fn load_bank_data(&self, data: &[u8])
    {
        self.load_preset_data(data)
    }
}