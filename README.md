ckmeans
=======

Ckmeans is a function to cluster a one-dimensional array of floats into `k` optimal clusters.
`Optimal` here means that the standard deviation inside each cluster is minimized.

Ckmeans was developed by
[Haizhou Wang and Mingzhou Song](http://journal.r-project.org/archive/2011-2/RJournal_2011-2_Wang+Song.pdf).

This code follows directly from Tom MacWright's code in
[simple-statistics](https://github.com/simple-statistics/simple-statistics/blob/master/src/ckmeans.js)
and David Schnurr's code in [d3-scale-cluster](https://github.com/schnerd/d3-scale-cluster/).

Please use this code in any way you wish, with or without attribution. It's licensed WTFPL.

If you'd rather use the original C++ code via a cython wrapper, check out
[rocketrip/ckmeans](https://github.com/rocketrip/ckmeans).

In 2022 ckmeans picked up a [citation](https://arxiv.org/ftp/arxiv/papers/2202/2202.04883.pdf)!
