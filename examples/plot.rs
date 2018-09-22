extern crate gnuplot;
extern crate meval;

use gnuplot::{Caption, Figure};
use meval::Expr;
use std::env::args;

const USAGE: &str = r"Plot functions of variable `x`.

Usage: plot EXPR1 EXPR2 ...

Example: plot 'sin(pi * x)'";

fn main() {
    let args = args().skip(1);
    if args.len() == 0 {
        println!("{}", USAGE);
    }

    let mut fg = Figure::new();
    fg.clear_axes();

    {
        let axes = fg.axes2d();

        let n = 100;
        let xi: Vec<_> = (0..n + 1).map(|i| i as f64 / n as f64).collect();

        for arg in args {
            // parse expression
            let expr = match arg.parse::<Expr>() {
                Ok(expr) => expr,
                Err(e) => return println!("Error when evaluating `{}`: {}", arg, e),
            };
            // create a function of one variable
            let func = match expr.bind("x") {
                Ok(func) => func,
                Err(e) => {
                    return println!("Error when trying to bind variable `x` in {}: {}", arg, e)
                }
            };

            axes.lines(
                &xi,
                xi.iter().map(|&x| func(x)),
                &[Caption("plot" /* &arg */)],
            );
        }
    }
    fg.show();
}
