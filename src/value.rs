use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, Copy)]
pub enum Op {
    Add,
    Mul,
    Tanh,
    None,
}

impl Op {
    fn backward(&self, out: &Value_, a: &ValueRef, b: &ValueRef) {
        match self {
            Op::Add => {
                if let Ok(mut a) = a.write() {
                    a.grad += 1.0 * out.grad;
                } else {
                    panic!("Failed to get reference to value");
                }
                if let Ok(mut b) = b.write() {
                    b.grad += 1.0 * out.grad;
                } else {
                    panic!("Failed to get reference to value");
                }
            }
            Op::Mul => {
                if let Ok(mut a) = a.write() {
                    a.grad += b.read().unwrap().data * out.grad;
                } else {
                    panic!("Failed to get reference to value");
                }
                if let Ok(mut b) = b.write() {
                    b.grad += a.read().unwrap().data * out.grad;
                } else {
                    panic!("Failed to get reference to value");
                }
            }
            Op::Tanh => {
                if let Ok(mut a) = a.write() {
                    let tanh = a.data.tanh();
                    a.grad += (1.0 - tanh * tanh) * out.grad;
                } else {
                    panic!("Failed to get reference to value");
                }
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

    fn backward(&self) {
        match self.prev.len() {
            1 => {
                self.op.backward(&self, &self.prev[0], &self.prev[0]);
                if let Ok(a) = self.prev[0].write() {
                    a.backward(); // recursively call backward on previous node
                } else {
                    panic!("Failed to get reference to value");
                }
            }
            2 => {
                self.op.backward(&self, &self.prev[0], &self.prev[1]);
                if let Ok(a) = self.prev[0].write() {
                    a.backward(); // recursively call backward on previous node
                } else {
                    panic!("Failed to get reference to value");
                }
                if let Ok(b) = self.prev[1].write() {
                    b.backward(); // recursively call backward on previous node
                } else {
                    panic!("Failed to get reference to value");
                }
            }
            _ => {}
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

    fn grad(&self) -> f64 {
        self.0.read().unwrap().grad
    }

    pub fn tanh(self) -> Self {
        Self::new_from_op(self.data().tanh(), Op::Tanh, vec![self.0.clone()])
    }

    pub fn backward(&self) {
        if let Ok(mut inner) = self.0.write() {
            inner.grad = 1.0;
        }
        if let Ok(inner) = self.0.read() {
            inner.backward();
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
    fn test_backward() {
        let x1 = Value::new(2.0);
        let x2 = Value::new(0.0);
        let w1 = Value::new(-3.0);
        let w2 = Value::new(1.0);
        let b = Value::new(6.8813735870195432);

        let x1w1 = x1.clone() * w1.clone();
        let x2w2 = x2.clone() * w2.clone();
        let x1w1x2w2 = x1w1.clone() + x2w2.clone();
        let n = x1w1x2w2.clone() + b;
        let o = n.clone().tanh();

        o.backward();

        assert!(float_eq(0.7071, o.data()));
        assert!(float_eq(0.5, n.grad()));
        assert!(float_eq(0.5, x2.grad()));
        assert!(float_eq(0.0, w2.grad()));
        assert!(float_eq(1.0, w1.grad()));
        assert!(float_eq(-1.5, x1.grad()));
    }

    #[test]
    fn test_gradient_accumulation() {
        let a = Value::new(3.0);
        let b = a.clone() + a.clone();
        b.backward();
        assert!(float_eq(2.0, a.grad()));

        let a = Value::new(-2.0);
        let b = Value::new(3.0);
        let d = a.clone() * b.clone();
        let e = a.clone() + b.clone();
        let f = d * e;
        f.backward();
        assert!(float_eq(-3.0, a.grad()));
        assert!(float_eq(-8.0, b.grad()));
    }
}
