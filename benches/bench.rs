#![feature(test)]

extern crate test;
extern crate meval;

use test::Bencher;
use meval::Expr;
use meval::HashContext;

const EXPR: &'static str = "abs(sin(x + 1) * (x^2 + x + 1))";

#[bench]
fn parsing(b: &mut Bencher) {
    b.iter(|| {
        Expr::from_str(EXPR).unwrap();
    });
}

#[bench]
fn evaluation(b: &mut Bencher) {
    let expr = Expr::from_str(EXPR).unwrap();
    let func = expr.bind("x").unwrap();
    b.iter(|| {
        func(1.);
    });
}

#[bench]
fn evaluation_hashcontext(b: &mut Bencher) {
    let expr = Expr::from_str(EXPR).unwrap();
    let func = expr.bind_with_context(HashContext::new(), "x").unwrap();
    b.iter(|| {
        func(1.);
    });
}
