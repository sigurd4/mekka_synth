use std::collections::BTreeSet;

use num::{rational::Ratio, Float, complex::ComplexFloat};
use serde_scala::{Scale, Pitch};

const C4: i32 = 44;

pub struct Tuning {
    pitches: Vec<Pitch>
}

impl Tuning
{
    pub fn from_scale(scale: Scale) -> Self
    {
        Self::from_pitches(scale.pitches)
    }

    pub fn from_pitches(pitches: Vec<Pitch>) -> Self
    {
        Self {
            pitches
        }
    }

    pub fn edo(edo: u32, octave: Ratio<u128>) -> Self
    {
        let octave_cents = Pitch::Ratio(octave).to_cents();
        let pitches: Vec<Pitch> = (0..edo).map(|n| (n + 1) as f64/edo as f64*octave_cents)
            .map(|cents| Pitch::Cents(cents))
            .collect();

        Tuning{
            pitches
        }
    }

    pub fn octave_note_count(&self) -> usize
    {
        self.pitches.len()
    }

    pub fn note(&self, note: i32) -> f64
    {
        let n_o = self.pitches.len() as i32;
        let mut n = note - C4 - 1;
        let c = n.div_floor(n_o);
        while n < 0
        {
            n += n_o;
        }
        while n >= n_o
        {
            n -= n_o;
        }
        let pitch = self.pitches[n as usize];
        let pitch_last = self.pitches.last().unwrap();

        (C4 + 1) as f64 + c as f64*pitch_last.to_note_offset() + pitch.to_note_offset()
    }
}