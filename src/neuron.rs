use rand::Rng;

use std::iter::IntoIterator;

use crate::value::Value;

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    Relu,
    Tanh,
    Linear,
}

pub struct Neuron {
    w: Vec<Value>,
    b: Value,
    act: Activation,
}

impl Neuron {
    pub fn new(nin: usize) -> Self {
        let mut rng = rand::thread_rng();
        let w = (0..nin)
            .map(|_| Value::new(rng.gen_range(-1.0..1.0)))
            .collect();
        let b = Value::new(rng.gen_range(-1.0..1.0));
        let act = Activation::Relu;
        Self { w, b, act }
    }

    pub fn with_activation(mut self, act: Activation) -> Self {
        self.act = act;
        self
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

        match self.act {
            Activation::Relu => act.relu(),
            Activation::Tanh => act.tanh(),
            Activation::Linear => act,
        }
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
    pub fn new(nin: usize, nout: usize, act: Activation) -> Self {
        let neurons = (0..nout)
            .map(|_| Neuron::new(nin).with_activation(act))
            .collect();
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
    pub fn new(nin: usize, mut nouts: Vec<usize>, act: Activation) -> Self {
        nouts.insert(0, nin);
        let mut layers = Vec::new();
        nouts
            .iter()
            .zip(nouts.iter().skip(1))
            .enumerate()
            .for_each(|(i, (&nin, &nout))| {
                let act = if i == nouts.len() - 2 {
                    Activation::Linear
                } else {
                    act
                };
                layers.push(Layer::new(nin, nout, act));
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

    pub fn zero_grad(&self) {
        self.parameters().iter().for_each(|p| p.zero_grad());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let x = vec![2.0, 3.0, -1.0];
        let n = MLP::new(3, vec![4, 4, 1], Activation::Tanh);
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
            let mlp = MLP::new(3, vec![4, 4, 1], Activation::Tanh);
            for k in 0..n {
                let ypred: Vec<Value> = xs.iter().flat_map(|x| mlp.run(x.clone())).collect();
                let loss = ys
                    .iter()
                    .zip(ypred.iter())
                    .fold(Value::new(0.0), |acc, (y, ypred)| {
                        acc + (y.clone() - ypred.clone()).pow(Value::new(2.0))
                    });

                mlp.zero_grad();
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
        println!("inital loss: {}, final loss: {}", initial_loss, final_loss);
        assert!(final_loss / initial_loss < 0.07);
    }

    fn loss(model: &MLP, data: &Vec<(f64, f64, i32)>, batch_size: Option<usize>) -> (Value, f64) {
        use rand::seq::SliceRandom;
        use rand::thread_rng;
        let batch_size = batch_size.unwrap_or(data.len());
        let batch: Vec<_> = data
            .choose_multiple(&mut thread_rng(), batch_size)
            .collect();

        let scores = batch
            .iter()
            .map(|&(a, b, _)| model.run(vec![Value::new(*a), Value::new(*b)]))
            .collect::<Vec<_>>();

        let losses = batch
            .iter()
            .map(|&(_, _, yb)| yb)
            .zip(scores.iter())
            .map(|(yi, scorei)| {
                let scorei = scorei[0].clone();
                (scorei * (-yi as f64) + 1.0).relu()
            })
            .collect::<Vec<_>>();

        let data_loss = losses
            .iter()
            .fold(Value::new(0.0), |sum, loss| sum + loss.clone())
            * (1.0 / losses.len() as f64);

        let alpha = 1.0e-4;
        let reg_loss = model
            .parameters()
            .iter()
            .map(|p| p.clone().pow(Value::new(2.0)))
            .fold(Value::new(0.0), |sum, p| sum + p)
            * alpha;
        let total_loss = data_loss + reg_loss;

        let result = batch
            .into_iter()
            .map(|&(_, _, yb)| yb)
            .zip(scores.iter())
            .map(|(yi, scorei)| (yi > 0) == (scorei[0].data() > 0.0))
            .collect::<Vec<_>>();

        let accuracy =
            result.iter().filter(|&&r| r).fold(0.0, |acc, _| acc + 1.0) / result.len() as f64;

        (total_loss, accuracy)
    }

    #[test]
    #[cfg(feature = "layout")]
    fn test_mlp_visualization() {
        let data = vec![(0.5, -0.2, 1)];
        let model = MLP::new(2, vec![2, 2, 1], Activation::Relu);
        for p in model.parameters().iter() {
            p.set_data(0.1);
        }
        let (total_loss, _) = loss(&model, &data, None);
        model.zero_grad();
        total_loss.backward();

        let mut vg = total_loss.trace();

        use layout::backends::svg::SVGWriter;
        use layout::core::utils::save_to_file;
        let mut svg = SVGWriter::new();
        vg.do_it(false, false, false, &mut svg);
        let _ = save_to_file("mlp_vis.svg", &svg.finalize());
    }

    #[test]
    fn test_binary_classifier() {
        for _ in 0..3 {
            let model = MLP::new(2, vec![16, 16, 1], Activation::Relu);
            let data = crate::moon::moon_data();

            let mut final_accuracy = 0.0;

            for k in 0..100 {
                let (total_loss, acc) = loss(&model, &data, None);
                model.zero_grad();
                total_loss.backward();

                let learning_rate = 1.0 - 0.9 * (k as f64) / 100.0;

                for p in model.parameters().iter() {
                    p.step(learning_rate);
                }

                if k % 1 == 0 {
                    println!(
                        "Step {:2.2} | loss {:3.3} | accuracy {:5.1}%\n",
                        k,
                        total_loss.data(),
                        acc * 100.0
                    );
                }
                final_accuracy = acc;
            }
            assert!(final_accuracy >= 0.99);
        }
    }
}
