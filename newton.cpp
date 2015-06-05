#include <iostream>
#include <cmath>

template <typename F, typename Fprime>
double newton(F f, Fprime fp, int iters, double x) {
    for(int i=0; i<iters; ++i) {
        auto dx = -f(x)/fp(x);
        x += dx;
    }

    return x;
}

double f(double x) {
    return std::exp(std::cos(x)) - 2.;
}

double fp(double x) {
    return -std::sin(x) * std::exp(std::cos(x));
}

int main(void) {
    auto root = newton(f, fp, 10, 5.);
    std::cout << "at root " << root << " f(x) = " << f(root) << std::endl;

    return 0;
}
