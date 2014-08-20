pylgssm
=======

Linear Gaussian state space models.

To Do
-----

- Make `*Base` classes abstract


Notes
-----

- For chain graphs we can store messages and marginals in arrays.  However, to
  extend to arbitrary graphs and loopy BP we should probably make a latent
  state class which stores pointers to its neighbors, its local
  potential/covariance, the covariance to its neighbors, and the outgoing
  message parameters.  We should be able to then compute messages efficiently
  with this.

- When doing minibatch stuff it's probably best to make copies of the
  minibatches to avoid weird race conditions.  Maybe this doesn't matter
  though.
