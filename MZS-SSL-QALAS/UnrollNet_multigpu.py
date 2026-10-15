import tensorflow as tf
import data_consistency as ssdu_dc
import tf_utils
import models.networks as networks
import parser_ops

parser = parser_ops.get_parser()
args = parser.parse_args()


class UnrolledNet():
    """

    Parameters
    ----------
    input_x: batch_size x nrow x ncol x 2
    sens_maps: batch_size x ncoil x nrow x ncol

    trn_mask: batch_size x nrow x ncol, used in data consistency units
    loss_mask: batch_size x nrow x ncol, used to define loss in k-space

    args.nb_unroll_blocks: number of unrolled blocks
    args.nb_res_blocks: number of residual blocks in ResNet

    Returns
    ----------

    x: nw output image
    nw_kspace_output: k-space corresponding nw output at loss mask locations

    x0 : dc output without any regularization.
    all_intermediate_results: all intermediate outputs of regularizer and dc units
    mu: learned penalty parameter


    """

    def __init__(self, input_x, sens_maps, trn_mask, loss_mask):
        self.input_x = input_x
        self.sens_maps = sens_maps
        self.trn_mask = trn_mask
        self.loss_mask = loss_mask
        self.model = self.Unrolled_SSDU()

    def Unrolled_SSDU(self):
        x, denoiser_output, dc_output = self.input_x, self.input_x, self.input_x
        all_intermediate_results = [[0 for _ in range(2)] for _ in range(args.nb_unroll_blocks)]

        mu_init = tf.constant(0., dtype=tf.float32)
        for ee in range(args.necho_GLOB):
            if ee == 0:
                x0 = ssdu_dc.dc_block(self.input_x[..., ee * 2:(ee + 1) * 2], self.sens_maps, self.trn_mask[..., ee], mu_init)
            else:
                x0 = tf.concat([x0, \
                        ssdu_dc.dc_block(self.input_x[..., ee * 2:(ee + 1) * 2], self.sens_maps, self.trn_mask[..., ee], mu_init)], axis=-1)
        x = x0


        sens_mask = tf.greater(tf.abs(self.sens_maps[:,0,:,:]), 0)
        sens_mask = tf.where(sens_mask, 1, 0)

        with tf.compat.v1.name_scope('SSDUModel'):
            with tf.compat.v1.variable_scope('Weights', reuse=tf.compat.v1.AUTO_REUSE):
                for i in range(args.nb_unroll_blocks):
                    if i == 0:
                        with tf.device("/gpu:1"):
                            x = networks.ResNet(x, args.nb_res_blocks, args.necho_GLOB)
                            denoiser_output = x

                            mu = networks.mu_param()
                            rhs = self.input_x + mu * x

                            for ee in range(args.necho_GLOB):
                                if ee == 0:
                                    x = ssdu_dc.dc_block(rhs[..., ee * 2:(ee + 1) * 2], self.sens_maps, self.trn_mask[..., ee], mu)
                                else:
                                    x = tf.concat([x, \
                                            ssdu_dc.dc_block(rhs[..., ee * 2:(ee + 1) * 2], self.sens_maps, self.trn_mask[..., ee], mu)], axis=-1)
                            x = x * tf.cast(tf.repeat(tf.expand_dims(sens_mask, -1), args.necho_GLOB*2, axis=3), tf.float32)
                            dc_output = x

                            # ...................................................................................................
                            all_intermediate_results[i][0] = tf_utils.tf_real2complex(tf.squeeze(denoiser_output))
                            all_intermediate_results[i][1] = tf_utils.tf_real2complex(tf.squeeze(dc_output))
                    elif i == 1:
                        with tf.device("/gpu:2"):
                            x = networks.ResNet(x, args.nb_res_blocks, args.necho_GLOB)
                            denoiser_output = x

                            mu = networks.mu_param()
                            rhs = self.input_x + mu * x

                            for ee in range(args.necho_GLOB):
                                if ee == 0:
                                    x = ssdu_dc.dc_block(rhs[..., ee * 2:(ee + 1) * 2], self.sens_maps, self.trn_mask[..., ee], mu)
                                else:
                                    x = tf.concat([x, \
                                            ssdu_dc.dc_block(rhs[..., ee * 2:(ee + 1) * 2], self.sens_maps, self.trn_mask[..., ee], mu)], axis=-1)
                            x = x * tf.cast(tf.repeat(tf.expand_dims(sens_mask, -1), args.necho_GLOB*2, axis=3), tf.float32)
                            dc_output = x

                            # ...................................................................................................
                            all_intermediate_results[i][0] = tf_utils.tf_real2complex(tf.squeeze(denoiser_output))
                            all_intermediate_results[i][1] = tf_utils.tf_real2complex(tf.squeeze(dc_output))
                    elif i == 2:
                        with tf.device("/gpu:3"):
                            x = networks.ResNet(x, args.nb_res_blocks, args.necho_GLOB)
                            denoiser_output = x

                            mu = networks.mu_param()
                            rhs = self.input_x + mu * x

                            for ee in range(args.necho_GLOB):
                                if ee == 0:
                                    x = ssdu_dc.dc_block(rhs[..., ee * 2:(ee + 1) * 2], self.sens_maps, self.trn_mask[..., ee], mu)
                                else:
                                    x = tf.concat([x, \
                                            ssdu_dc.dc_block(rhs[..., ee * 2:(ee + 1) * 2], self.sens_maps, self.trn_mask[..., ee], mu)], axis=-1)
                            x = x * tf.cast(tf.repeat(tf.expand_dims(sens_mask, -1), args.necho_GLOB*2, axis=3), tf.float32)
                            dc_output = x

                            # ...................................................................................................
                            all_intermediate_results[i][0] = tf_utils.tf_real2complex(tf.squeeze(denoiser_output))
                            all_intermediate_results[i][1] = tf_utils.tf_real2complex(tf.squeeze(dc_output))
                    elif i == 3:
                        with tf.device("/gpu:4"):
                            x = networks.ResNet(x, args.nb_res_blocks, args.necho_GLOB)
                            denoiser_output = x

                            mu = networks.mu_param()
                            rhs = self.input_x + mu * x

                            for ee in range(args.necho_GLOB):
                                if ee == 0:
                                    x = ssdu_dc.dc_block(rhs[..., ee * 2:(ee + 1) * 2], self.sens_maps, self.trn_mask[..., ee], mu)
                                else:
                                    x = tf.concat([x, \
                                            ssdu_dc.dc_block(rhs[..., ee * 2:(ee + 1) * 2], self.sens_maps, self.trn_mask[..., ee], mu)], axis=-1)
                            x = x * tf.cast(tf.repeat(tf.expand_dims(sens_mask, -1), args.necho_GLOB*2, axis=3), tf.float32)
                            dc_output = x

                            # ...................................................................................................
                            all_intermediate_results[i][0] = tf_utils.tf_real2complex(tf.squeeze(denoiser_output))
                            all_intermediate_results[i][1] = tf_utils.tf_real2complex(tf.squeeze(dc_output))
                    elif i == 4:
                        with tf.device("/gpu:1"):
                            x = networks.ResNet(x, args.nb_res_blocks, args.necho_GLOB)
                            denoiser_output = x

                            mu = networks.mu_param()
                            rhs = self.input_x + mu * x

                            for ee in range(args.necho_GLOB):
                                if ee == 0:
                                    x = ssdu_dc.dc_block(rhs[..., ee * 2:(ee + 1) * 2], self.sens_maps, self.trn_mask[..., ee], mu)
                                else:
                                    x = tf.concat([x, \
                                            ssdu_dc.dc_block(rhs[..., ee * 2:(ee + 1) * 2], self.sens_maps, self.trn_mask[..., ee], mu)], axis=-1)
                            x = x * tf.cast(tf.repeat(tf.expand_dims(sens_mask, -1), args.necho_GLOB*2, axis=3), tf.float32)
                            dc_output = x

                            # ...................................................................................................
                            all_intermediate_results[i][0] = tf_utils.tf_real2complex(tf.squeeze(denoiser_output))
                            all_intermediate_results[i][1] = tf_utils.tf_real2complex(tf.squeeze(dc_output))
                    elif i == 5:
                        with tf.device("/gpu:2"):
                            x = networks.ResNet(x, args.nb_res_blocks, args.necho_GLOB)
                            denoiser_output = x

                            mu = networks.mu_param()
                            rhs = self.input_x + mu * x

                            for ee in range(args.necho_GLOB):
                                if ee == 0:
                                    x = ssdu_dc.dc_block(rhs[..., ee * 2:(ee + 1) * 2], self.sens_maps, self.trn_mask[..., ee], mu)
                                else:
                                    x = tf.concat([x, \
                                            ssdu_dc.dc_block(rhs[..., ee * 2:(ee + 1) * 2], self.sens_maps, self.trn_mask[..., ee], mu)], axis=-1)
                            x = x * tf.cast(tf.repeat(tf.expand_dims(sens_mask, -1), args.necho_GLOB*2, axis=3), tf.float32)
                            dc_output = x

                            # ...................................................................................................
                            all_intermediate_results[i][0] = tf_utils.tf_real2complex(tf.squeeze(denoiser_output))
                            all_intermediate_results[i][1] = tf_utils.tf_real2complex(tf.squeeze(dc_output))
                    elif i == 6:
                        with tf.device("/gpu:3"):
                            x = networks.ResNet(x, args.nb_res_blocks, args.necho_GLOB)
                            denoiser_output = x

                            mu = networks.mu_param()
                            rhs = self.input_x + mu * x

                            for ee in range(args.necho_GLOB):
                                if ee == 0:
                                    x = ssdu_dc.dc_block(rhs[..., ee * 2:(ee + 1) * 2], self.sens_maps, self.trn_mask[..., ee], mu)
                                else:
                                    x = tf.concat([x, \
                                            ssdu_dc.dc_block(rhs[..., ee * 2:(ee + 1) * 2], self.sens_maps, self.trn_mask[..., ee], mu)], axis=-1)
                            x = x * tf.cast(tf.repeat(tf.expand_dims(sens_mask, -1), args.necho_GLOB*2, axis=3), tf.float32)
                            dc_output = x

                            # ...................................................................................................
                            all_intermediate_results[i][0] = tf_utils.tf_real2complex(tf.squeeze(denoiser_output))
                            all_intermediate_results[i][1] = tf_utils.tf_real2complex(tf.squeeze(dc_output))
                    elif i == 7:
                        with tf.device("/gpu:4"):
                            x = networks.ResNet(x, args.nb_res_blocks, args.necho_GLOB)
                            denoiser_output = x

                            mu = networks.mu_param()
                            rhs = self.input_x + mu * x

                            for ee in range(args.necho_GLOB):
                                if ee == 0:
                                    x = ssdu_dc.dc_block(rhs[..., ee * 2:(ee + 1) * 2], self.sens_maps, self.trn_mask[..., ee], mu)
                                else:
                                    x = tf.concat([x, \
                                            ssdu_dc.dc_block(rhs[..., ee * 2:(ee + 1) * 2], self.sens_maps, self.trn_mask[..., ee], mu)], axis=-1)
                            x = x * tf.cast(tf.repeat(tf.expand_dims(sens_mask, -1), args.necho_GLOB*2, axis=3), tf.float32)
                            dc_output = x

                            # ...................................................................................................
                            all_intermediate_results[i][0] = tf_utils.tf_real2complex(tf.squeeze(denoiser_output))
                            all_intermediate_results[i][1] = tf_utils.tf_real2complex(tf.squeeze(dc_output))
                    else:
                        with tf.device("/gpu:0"):
                            x = networks.ResNet(x, args.nb_res_blocks, args.nbasis_GLOB)
                            denoiser_output = x

                            mu = networks.mu_param()
                            rhs = self.input_x + mu * x

                            x = ssdu_dc.dc_block(rhs, self.sens_maps, self.trn_mask, self.basis, self.stk, mu)
                            # x = rhs # to check w/o CG
                            dc_output = x

                            # ...................................................................................................
                            for kk in range(args.nbasis_GLOB):
                                denoiser_output_c_ = tf.squeeze(denoiser_output)
                                denoiser_output_c_ = tf.expand_dims(tf_utils_subspace.tf_real2complex(denoiser_output_c_[..., kk * 2:(kk + 1) * 2]), axis=-1)
                                dc_output_c_ = tf.squeeze(dc_output)
                                dc_output_c_ = tf.expand_dims(tf_utils_subspace.tf_real2complex(dc_output_c_[..., kk * 2:(kk + 1) * 2]), axis=-1)
                                if kk == 0:
                                    denoiser_output_c = denoiser_output_c_
                                    dc_output_c = dc_output_c_
                                else:
                                    denoiser_output_c = tf.concat([denoiser_output_c, denoiser_output_c_], axis=-1)
                                    dc_output_c = tf.concat([dc_output_c, dc_output_c_], axis=-1)
                            all_intermediate_results[i][0] = denoiser_output_c
                            all_intermediate_results[i][1] = dc_output_c




            for ee in range(args.necho_GLOB):
                if ee == 0:
                    nw_kspace_output = ssdu_dc.SSDU_kspace_transform(x[..., ee * 2:(ee + 1) * 2], self.sens_maps, self.loss_mask[..., ee])
                else:
                    nw_kspace_output = tf.concat([nw_kspace_output, \
                                            ssdu_dc.SSDU_kspace_transform(x[..., ee * 2:(ee + 1) * 2], self.sens_maps, self.loss_mask[..., ee])], axis=-1)
        
        return x, nw_kspace_output, x0, all_intermediate_results, mu
