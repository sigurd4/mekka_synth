use crate::parameters::EnvelopeParameters;

#[derive(Clone, Copy)]
pub struct ADSREnvelope
{
    pub i: Option<usize>,
    pub h: Option<usize>,
    pub a: f64,
    pub d: f64,
    pub s: f64,
    pub r: f64
}

impl ADSREnvelope
{
    pub fn new() -> Self
    {
        Self {
            i: None,
            h: None,
            a: 0.0,
            d: 0.0,
            s: 1.0,
            r: 0.0
        }
    }

    pub fn is_held(&self) -> bool
    {
        self.i.is_some() && self.h.is_none()
    }

    pub fn next(&mut self, rate: f64) -> f64
    {
        let e = self.envelope(rate);
        self.i = self.i.and_then(|i| if i == usize::MAX {None} else {Some(i + 1)});
        return e
    }

    pub fn envelope(&self, rate: f64) -> f64
    {
        match self.i
        {
            Some(i) => {
                let t = i as f64/rate;
                let t_h = match self.h
                {
                    Some(h) => h as f64/rate,
                    None => f64::INFINITY
                };
        
                if t <= 0.0
                {
                    return 0.0
                }
        
                let mut e = 1.0;
        
                if t < self.a
                {
                    e *= (t.min(t_h)/self.a).exp2() - 1.0;
                }
                else
                {
                    if self.a < t_h
                    {
                        e *= self.s + (1.0 - self.s)*((self.a - t)/self.d).exp2()
                    }
                    else
                    {
                        e *= (t_h/self.a).exp2() - 1.0
                    }
                }
                if t > t_h
                {
                    e *= ((t_h - t)/self.r).exp2()
                }
        
                return e
            },
            None => 0.0
        }
    }

    pub fn set_parameters(&mut self, params: &EnvelopeParameters)
    {
        self.a = params.attack.get() as f64;
        self.d = params.decay.get() as f64;
        self.s = params.sustain.get() as f64;
        self.r = params.release.get() as f64;
    }
}