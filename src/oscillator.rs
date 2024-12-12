use std::f64::consts::{TAU, PI};

use crate::waveform::{Waveform, MAX_SERIES};

const TUNE: f64 = 0.0;

#[derive(Clone)]
pub struct Oscillator
{
    pub note: f64,
    waveform: Waveform,
    duty_cycle: f64,
    wavetable: Box<[(f64, f64); MAX_SERIES]>,
    pub theta: f64
}

impl Oscillator
{
    pub fn new(note: f64, waveform: Waveform, duty_cycle: f64) -> Self
    {
        Self {
            note,
            waveform,
            duty_cycle,
            wavetable: Box::new(waveform.wavetable(duty_cycle)),
            theta: 0.0
        }
    }

    pub fn omega(&self) -> f64
    {
        TAU*440.0*((self.note - 49.0 + TUNE)/12.0).exp2()
    }

    pub fn step(&mut self, rate: f64)
    {
        let omega_norm = self.omega()/rate*self.waveform.omega_mul();
        self.theta = (self.theta + omega_norm) % TAU;
    }

    pub fn next(&mut self, rate: f64) -> f64
    {
        let omega_norm = self.omega()/rate*self.waveform.omega_mul();
        let y = self.waveform(omega_norm);
        self.step(rate);
        y
    }

    pub fn set_waveform(&mut self, waveform: Waveform, duty_cycle: f64)
    {
        if self.waveform != waveform || self.duty_cycle != duty_cycle
        {
            self.wavetable = Box::new(waveform.wavetable(duty_cycle));
        }
        self.duty_cycle = duty_cycle;
        self.waveform = waveform;
    }

    fn waveform(&self, omega_norm: f64) -> f64
    {
        let n_max = (PI/omega_norm).ceil() as usize;
        if n_max >= MAX_SERIES
        {
            return self.waveform.waveform_direct(self.theta, self.duty_cycle);
        }
        let n_max = n_max.min(MAX_SERIES);
        match self.waveform
        {
            Waveform::Sine => {
                Waveform::waveform_wavetable(&self.wavetable[..n_max], self.theta)
            },
            Waveform::Triangle => {
                Waveform::waveform_wavetable(&self.wavetable[..n_max], self.theta)
            },
            Waveform::Sawtooth => {
                Waveform::waveform_wavetable(&self.wavetable[..n_max], self.theta)
            },
            Waveform::Square => {
                Waveform::waveform_wavetable(&self.wavetable[..n_max], self.theta)
            },
            Waveform::Noise => {
                self.waveform.waveform_direct(self.theta, self.duty_cycle)
            },
        }
    }
}
