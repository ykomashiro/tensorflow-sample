import numpy as np
"""
There are some layers for recurrent neural network.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Inputs:
        - x: input data for this timetamp, of shape (N, D)
        - prev_h: hidden state from previous timetamp, of shape (N, H)
        - Wx: weight matrix for input-to-hidden, of shape (D, H)
        - Wb: weight matrix for hidden-to-hidden,of shape (H, H)
        - b: biases, of shape (H,)
    Returns:
        - next_h: next hidden state, of shape (H, H)
        - cache: tuple of values that used for backword step.
    """
    next_h, cache = None, None
    next_h = np.tanh(x.dot(Wx) + prev_h.dot(Wh) + b)
    cache = (x, prev_h, Wh, Wx, b, next_h)
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Inputs:
        - dnext_h: gradient of loss with respect to the next timetamp.
        - cache: from forword pass.
    Returns:
        - dx: gradient of input data, of shape (N, D)
        - dprev_h: gradient of previous hiddent state, of shape (N, H)
        - dWx: of shape (D, H)
        - dWh: of shape (H, H)
        - db: of shape (H,)
    """
    dx, dprev_h, dWh, dWx, db = None, None, None, None, None
    (x, prev_h, Wh, Wx, b, next_h) = cache
    # middle unit of shape (N, H).
    dtheta = dnext_h * (1 - next_h**2)

    db = np.sum(dtheta, axis=0)
    dprev_h = dtheta.dot(Wh.T)
    dWx = prev_h.T.dot(dtheta)
    dWh = x.T.dot(dtheta)
    dx = dtheta.dot(Wx.T)
    return dx, dprev_h, dWx, dWh, db


def rnn_forword(x, h0, Wx, Wh, b):
    """
    Inputs:
        - x: a series of input data, of shape (N, T, D)
        - h0: initial hidden state, of shape (N, H)
        - Wx: weight matrix for input-to-hidden, of shape (D, H)
        - Wb: weight matrix for hidden-to-hidden,of shape (H, H)
        - b: biases, of shape (H,)
    Returns:
        - h: a series of hidden states, of shape (N, T, H)
        - cache: tuple of values that used for backword step.
    """
    N, T, D = x.shape
    H = h0.shape[1]
    prev_h = h0
    h = np.zeros((N, T, H))
    for t in range(T):
        xt = x[:, t, :]
        xt = xt.reshape(N, D)
        next_h, cache = rnn_step_forward(x, prev_h, Wx, Wh, b)
        h[:, t, :] = next_h
        prev_h = next_h
    cache = (x, h0, Wh, Wx, b, h)
    return h, cache


def rnn_backword(dh, cache):
    """
    Inputs:
        - dh: upstream gradient from all hidden states, of shape (N,T,H)
        - cache: tuple of values that used for backword step.
    Returns:
        - dx:
        - dh0:
        - dWh:
        - dWx:
        - db:
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    (x, h0, Wh, Wx, b, h) = cache
    N, T, D = x.shape
    H = h0.shape[1]
    dprev_h = np.zeros((N, H))
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, H))
    dWh = np.zeros((H, H))
    db = np.zeros((H, ))
    for t in range(T - 1, -1, -1):
        if t > 0:
            prev_h = h[:, t - 1, :]
        else:
            prev_h = h0
        xt = x[:, t, :]
        xt = xt.reshape(N, D)
        next_h = h[:, t, :]
        cache = (xt, prev_h, Wh, Wx, b, next_h)
        dnext_h = dh[:, t, :] + dprev_h
        dx[:, t, :], dprev_h, dWxt, dWht, dbt = rnn_step_backward(
            dnext_h, cache)
        dWx += dWxt
        dWh += dWht
        db += dbt
        dh0 = dprev_h
        return dx, dh0, dWx, dWh, db
