def test_TH_loss():
  # TODO add imports and expected results


  offsets_np = np.array([[[0.1, 0.40, 0.50], [0.1, 0.40, 0.50], [0.1, 0.40, 0.50]],  # segment 1
                         [[0.20, 0.20, 0.60], [0.20, 0.20, 0.60], [0.20, 0.20, 0.60]]])  # segment 2
  targets_enc_np = np.array([[1, 1, 1],  # frame 1
                             [2, 2, 2],
                             [3, 3, 3],
                             [4, 4, 4]])[:, :, None]
  targets_img_np = targets_enc_np[:, :, :, None, None]

  offsets = tf.placeholder(tf.float32, shape=offsets_np.shape)
  targets_enc = tf.placeholder(tf.float32, shape=targets_enc_np.shape)
  targets_img = tf.placeholder(tf.float32, shape=targets_img_np.shape)
  targets_res, weights_res, propagated_distributions = get_high_level_targets(offsets, targets_enc)
  targets_low, weights_low = get_low_level_targets(offsets, targets_img, propagated_distributions)

  with tf.Session() as sess:
    res = sess.run([targets_res, weights_res, propagated_distributions, targets_low, weights_low],
                   {offsets: offsets_np, targets_enc: targets_enc_np, targets_img: targets_img_np})