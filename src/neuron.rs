use rand::Rng;

use std::iter::IntoIterator;

use crate::value::Value;

pub struct Neuron {
    w: Vec<Value>,
    b: Value,
}

impl Neuron {
    pub fn new(nin: usize) -> Self {
        let mut rng = rand::thread_rng();
        let w = (0..nin)
            .map(|_| Value::new(rng.gen_range(-1.0..1.0)))
            .collect();
        let b = Value::new(rng.gen_range(-1.0..1.0));
        Self { w, b }
    }
    pub fn run<T, U>(&self, x: T) -> Value
    where
        T: IntoIterator<Item = U>,
        U: Into<Value>,
    {
        let act = self
            .w
            .iter()
            .zip(x.into_iter())
            .fold(self.b.clone(), |acc, (w, x)| acc + w.clone() * x.into());
        let out = act.tanh();
        out
    }
    pub fn parameters(&self) -> Vec<Value> {
        let mut params = self.w.clone();
        params.push(self.b.clone());
        params
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Self {
        let neurons = (0..nout).map(|_| Neuron::new(nin)).collect();
        Self { neurons }
    }

    pub fn run<T, U>(&self, x: T) -> Vec<Value>
    where
        T: IntoIterator<Item = U> + Clone,
        U: Into<Value>,
    {
        self.neurons.iter().map(|n| n.run(x.clone())).collect()
    }

    pub fn parameteres(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: usize, mut nouts: Vec<usize>) -> Self {
        nouts.insert(0, nin);
        let mut layers = Vec::new();
        nouts
            .iter()
            .zip(nouts.iter().skip(1))
            .for_each(|(&nin, &nout)| {
                layers.push(Layer::new(nin, nout));
            });

        Self { layers }
    }

    pub fn run<T, U>(&self, x: T) -> Vec<Value>
    where
        T: IntoIterator<Item = U> + Clone,
        U: Into<Value>,
    {
        let mut x = x.into_iter().map(|v| v.into()).collect::<Vec<Value>>();
        for layer in self.layers.iter() {
            x = layer.run(x);
        }
        x
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|l| l.parameteres()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        let x = vec![2.0, 3.0, -1.0];
        let n = MLP::new(3, vec![4, 4, 1]);
        let r = n.run(x);
        println!("{:?}", r.iter().map(|v| v.data()).collect::<Vec<f64>>());
    }

    #[test]
    fn test_gradient_descent() {
        let xs = vec![
            vec![2.0, 3.0, -1.0],
            vec![3.0, -1.0, 0.5],
            vec![0.5, 1.0, 1.0],
            vec![1.0, 1.0, -1.0],
        ];
        let ys = vec![
            Value::new(1.0),
            Value::new(-1.0),
            Value::new(-1.0),
            Value::new(1.0),
        ];

        let n = 20;

        let mut init = Vec::new();
        let mut fin = Vec::new();

        use std::time::Instant;

        let start = Instant::now();

        for _ in 0..1000 {
            let mlp = MLP::new(3, vec![4, 4, 1]);
            for k in 0..n {
                let ypred: Vec<Value> = xs.iter().flat_map(|x| mlp.run(x.clone())).collect();
                let loss = ys
                    .iter()
                    .zip(ypred.iter())
                    .fold(Value::new(0.0), |acc, (y, ypred)| {
                        acc + (y.clone() - ypred.clone()).pow(Value::new(2.0))
                    });
                loss.backward();
                if k == 0 {
                    init.push(loss.data());
                }
                if k == n - 1 {
                    fin.push(loss.data());
                }

                for p in mlp.parameters().iter() {
                    p.step(0.05);
                }
            }
        }
        let _elapsed = start.elapsed().as_millis();
        let initial_loss = init.iter().sum::<f64>() / init.len() as f64;
        let final_loss = fin.iter().sum::<f64>() / fin.len() as f64;
        assert!(final_loss / initial_loss < 0.1);
    }
}
