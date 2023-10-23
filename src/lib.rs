#![allow(dead_code)]

pub mod moon;
pub mod neuron;
pub mod util;
pub mod value;

pub mod prelude {
    pub use crate::moon;
    pub use crate::neuron::{Activation, Layer, Neuron, MLP};
    pub use crate::value::Value;
}
