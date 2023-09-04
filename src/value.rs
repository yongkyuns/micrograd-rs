use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, Copy)]
pub enum Op {
    Add,
    Mul,
    Tanh,
    None,
}

impl Op {
    fn backward(&self, out: &Value_, a: &mut Value_, b: &mut Value_) {
        match self {
            Op::Add => {
                a.grad += 1.0 * out.grad;
                b.grad += 1.0 * out.grad;
            }
            Op::Mul => {
                a.grad += b.data * out.grad;
                b.grad += a.data * out.grad;
            }
            Op::Tanh => {
                let tanh = a.data.tanh();
                a.grad += (1.0 - tanh * tanh) * out.grad;
            }
            Op::None => {}
        }
    }
}

#[derive(Debug, Clone)]
struct Value_ {
    data: f64,
    prev: Vec<ValueRef>,
    op: Op,
    grad: f64,
}

impl Value_ {
    fn new(data: f64) -> Self {
        Self {
            data,
            op: Op::None,
            prev: vec![],
            grad: 0.0,
        }
    }
}

type ValueRef = Arc<RwLock<Value_>>;

#[derive(Debug, Clone)]
pub struct Value(ValueRef);

impl Value {
    pub fn new(data: f64) -> Self {
        Self(Arc::new(RwLock::new(Value_::new(data))))
    }

    fn new_from_op(data: f64, op: Op, prev: Vec<ValueRef>) -> Self {
        Self(Arc::new(RwLock::new(Value_ {
            data,
            op,
            prev,
            grad: 0.0,
        })))
    }

    fn data(&self) -> f64 {
        self.0.read().unwrap().data
    }

    fn set_data(&mut self, data: f64) {
        self.0.write().unwrap().data = data;
    }

    fn grad(&self) -> f64 {
        self.0.read().unwrap().grad
    }

    fn set_grad(&mut self, grad: f64) {
        self.0.write().unwrap().grad = grad;
    }

    fn prev(&self) -> Vec<ValueRef> {
        self.0.read().unwrap().prev.clone()
    }

    fn op(&self) -> Op {
        self.0.read().unwrap().op
    }

    pub fn tanh(self) -> Self {
        Self::new_from_op(self.data().tanh(), Op::Tanh, vec![self.0.clone()])
    }

    pub fn backward(&self) {
        match self.prev().len() {
            1 => {
                if let (Ok(s), Ok(mut a)) = (self.0.read(), self.prev()[0].write()) {
                    self.op().backward(&s, &mut a, &mut Value_::new(0.0));
                } else {
                    panic!("Failed to get reference to value");
                }
            }
            2 => {
                if let (Ok(s), Ok(mut a), Ok(mut b)) = (
                    self.0.read(),
                    self.prev()[0].write(),
                    self.prev()[1].write(),
                ) {
                    self.op().backward(&s, &mut a, &mut b);
                } else {
                    panic!("Failed to get reference to value");
                }
            }
            _ => {}
        }
    }
}

impl std::ops::Add for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Value {
        Self::new_from_op(
            self.data() + rhs.data(),
            Op::Add,
            vec![self.0.clone(), rhs.0.clone()],
        )
    }
}

impl std::ops::Mul for Value {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::new_from_op(
            self.data() * rhs.data(),
            Op::Mul,
            vec![self.0.clone(), rhs.0.clone()],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_case() {
        let h = 0.001;

        let a = Value::new(2.0);
        let b = Value::new(-3.0);
        let c = Value::new(10.0);
        let e = a * b;
        let d = e + c;
        let f = Value::new(-2.0);
        let L = d * f;
        let L1 = L.data();

        let a = Value::new(2.0);
        let b = Value::new(-3.0);
        let c = Value::new(10.0);
        let e = a * b;
        let d = e + c;
        let f = Value::new(-2.0);
        let L = d * f;
        let L2 = L.data();
        println!("{:?}", (L2 - L1) / h);

        let x1 = Value::new(2.0);
        let x2 = Value::new(0.0);
        let w1 = Value::new(-3.0);
        let w2 = Value::new(1.0);
        let b = Value::new(6.7);
        let x1w1 = x1 * w1;
        let x2w2 = x2 * w2;
        let x1w1x2w2 = x1w1 + x2w2;
        let n = x1w1x2w2 + b;
        let o = n.tanh();
        println!("{}", o.data());
    }

    fn float_eq(exp: f64, act: f64) -> bool {
        if exp < std::f64::EPSILON {
            if act < std::f64::EPSILON {
                true
            } else {
                false
            }
        } else {
            let tol = exp * 1.0e-4;
            (exp - act).abs() < tol
        }
    }

    #[test]
    fn test_tanh() {
        let x1 = Value::new(2.0);
        let x2 = Value::new(0.0);
        let w1 = Value::new(-3.0);
        let w2 = Value::new(1.0);
        let b = Value::new(6.8813735870195432);

        let x1w1 = x1.clone() * w1.clone();
        let x2w2 = x2.clone() * w2.clone();
        let x1w1x2w2 = x1w1.clone() + x2w2.clone();
        let n = x1w1x2w2.clone() + b;
        let mut o = n.clone().tanh();

        o.set_grad(1.0);
        o.backward();
        n.backward();
        x1w1x2w2.backward();
        x2w2.backward();
        x1w1.backward();

        assert!(float_eq(0.7071, o.data()));
        assert!(float_eq(0.5, n.grad()));
        assert!(float_eq(0.5, x2.grad()));
        assert!(float_eq(0.0, w2.grad()));
        assert!(float_eq(1.0, w1.grad()));
        assert!(float_eq(-1.5, x1.grad()));
    }
}
