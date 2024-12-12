use std::f64::EPSILON;
use std::f64::consts::{PI, FRAC_2_PI, TAU};

use num::Complex;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::thread_rng;

pub const MAX_SERIES: usize = 4096;

#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(u8)]
pub enum Waveform
{
    Sine,
    Triangle,
    Sawtooth,
    Square,
    Noise
}

impl Waveform
{
    pub const VARIANT_COUNT: usize = core::mem::variant_count::<Self>();
    pub const WAVEFORMS: [Self; Self::VARIANT_COUNT] = [Self::Sine, Self::Triangle, Self::Sawtooth, Self::Square, Self::Noise];

    pub fn omega_mul(&self) -> f64
    {
        match self
        {
            Waveform::Sine => 1.0,
            Waveform::Triangle => 1.0,
            Waveform::Sawtooth => 0.5,
            Waveform::Square => 1.0,
            Waveform::Noise => 1.0,
        }
    }

    pub fn wavetable(&self, duty_cycle: f64) -> [(f64, f64); MAX_SERIES]
    {
        let d = TAU*duty_cycle.max(0.0).min(1.0);
        match self
        {
            Waveform::Sine => core::array::from_fn(|m| {
                if d == PI
                {
                    return (0.0, if m == 0 {1.0} else {0.0})
                }
                let n = m + 1;
                let g = 1.0/n as f64/PI*(1.0/((1.0 - PI/n as f64/d)*(1.0 + PI/n as f64/d) + EPSILON) - 1.0/((1.0 + PI/n as f64/(d - TAU))*(1.0 - PI/n as f64/(d - TAU)) + EPSILON));
                if g.is_nan()
                {
                    return (0.0, 0.0)
                }
                let dn = d*n as f64;
                (
                    g*dn.sin(),
                    -g*(dn.cos() + 1.0)
                )
            }),
            Waveform::Triangle => core::array::from_fn(|m| {
                let n = m + 1;
                if d == 0.0
                {
                    return (0.0, 2.0/(n as f64)/PI)
                }
                if d == TAU
                {
                    return (0.0, -2.0/(n as f64)/PI)
                }
                let g = 4.0/((n*n) as f64)/(TAU - d + EPSILON)/(d + EPSILON);
                let dn = d*n as f64;
                (
                    g*(dn.cos() - 1.0),
                    g*dn.sin()
                )
            }),
            Waveform::Sawtooth => core::array::from_fn(|m| {
                let n = m + 1;
                if d == 0.0 || d == TAU
                {
                    return (0.0, -2.0/(n as f64)/PI)
                }
                let n = m + 1;
                let g = 1.0/n as f64*(1.0/d-1.0/(TAU-d));
                if g.is_nan()
                {
                    return (0.0, 0.0)
                }
                let dn = d*n as f64;
                (
                    FRAC_2_PI/n as f64*(g*(dn.cos() - 1.0) + dn.sin()),
                    FRAC_2_PI/n as f64*(g*dn.sin() - 1.0 - dn.cos())
                )
            }),
            Waveform::Square => core::array::from_fn(|m| {
                let n = m + 1;
                let g = 2.0/(PI*n as f64);
                if g.is_nan()
                {
                    return (0.0, 0.0)
                }
                let dn = d*n as f64;
                (
                    -g*dn.sin(),
                    g*(dn.cos() - 1.0)
                )
            }),
            Waveform::Noise => [(0.0, 0.0); MAX_SERIES]
        }
    }

    pub fn waveform_direct(&self, theta: f64, duty_cycle: f64) -> f64
    {
        let d = (TAU*duty_cycle) % TAU;
        let theta = theta % TAU;
        match self
        {
            Waveform::Sine => {
                theta.sin()
            },
            Waveform::Triangle => {
                if theta < d
                {
                    2.0*theta/d - 1.0
                }
                else
                {
                    1.0 - 2.0*(theta - d)/(TAU - d)
                }
            },
            Waveform::Sawtooth => {
                if theta < d
                {
                    2.0*theta/d - 1.0
                }
                else
                {
                    2.0*(theta - d)/(TAU - d) - 1.0
                }
            },
            Waveform::Square => {
                return if theta < PI {1.0} else {-1.0}
            },
            Waveform::Noise => {
                let y: f64 = Standard.sample(&mut thread_rng());
                return y.max(-1.0).min(1.0)
            },
        }
    }

    pub fn waveform_wavetable(wavetable: &[(f64, f64)], theta: f64) -> f64
    {
        let mut y = 0.0;
        let exp_1 = Complex::cis(theta);
        let mut exp_n = exp_1;
        for (a, b) in wavetable.iter()
        {
            y += a*exp_n.re + b*exp_n.im;
            exp_n *= exp_1;
        }
        return y
    }
}