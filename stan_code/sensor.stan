data {
    matrix[2, 2] x_known;                   // Two known sensor positions
    matrix[6, 6] y;                         // Knwon distances
    array[6, 6] int<lower=0, upper=1> w;    // Indicator if distance is known
}

parameters {
    matrix[4, 2] x_unknown;                 // Four unknown sensor positions
}

model {
    for (i in 1:6) {
        for (j in (i+1):6) {
            row_vector[2] xi = i <= 4 ? x_unknown[i] : x_known[i - 4];
            row_vector[2] xj = j <= 4 ? x_unknown[j] : x_known[j - 4];
            real d  = sqrt((xi[1] - xj[1])^2 + (xi[2] - xj[2])^2);
            target += w[i][j] * normal_lpdf(y[i][j] | d, 0.02) + bernoulli_lpmf(w[i][j] | exp(-0.5 * d^2 / 0.3^2));
        }
    }
}