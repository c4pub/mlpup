# >-----------------------------------------------------------------------------
# >-----------------------------------------------------------------------------

#   c4pub@git 2025
#
# Latest version available at: https://github.com/c4pub/deodel
#

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

"""
This is a one-file implementation of a Multi-Layer Perceptron (MLP).
It is intended as a basic implementation of the backpropagation
algorithm and does not attempt any performance optimization.

The first part contains comments describing a theoretical derivation for
backpropagation. The second part contains code that implements the
theoretical description.

Several classes are used, but mostly for encapsulation purposes.
Many of the classes implement generic utility functions, e.g., to format
the inputs fed to the classifier and the translation of outputs.
The class that contains the actual implementation of the core
functionality is NeuNet. Cues to the implementation of the theoretical
concepts are given by the <theory-ref> keyword comments.

Features:
  - One-file, self-contained implementation
  - No external libraries are used for the implementation of the core
      functionality. NumPy and others are used only for preparing the
      feed data.
  - Compatibility with classifier usage in scikit-learn
  - Accepts a mix of continuous and categorical data
  - Several activation and loss functions are provided.
  - Can also do basic regression

"""

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


'''

    (*) Theoretical background on back-propagation and weights update.

        Gradient descent: minimizing a loss function f_loss(x) by varying x so
        that it descends in small steps towards the minimum. The theory says
        that the most efficient update is:

        x_new = x_crt - rate_step * (d(f_loss(x))/d(x))(x_crt)

        (*) where:

            rate_step = learn_rate

        (*) The term "gradient" refers to the derivatives of the loss function
            with respect to each of the weight components, treating these
            weights as a vector:

                    d(f_loss(x))/d(x) <=> grad(f_loss(x))

    (*) In back-propagation:

        w_new = w_crt - learn_rate * (d(f_loss(w))/d(w))(w_crt)

        (*) The difference between the predictive and the expected output
            represents the residual.

                residual = y_out - y_expect
                residual(w, in) = f_out(w, in) - f_expect(in)

                    f_out() - also referred to as activation

        (*) Definitions:

            w - weights,
            in - inputs,
            f_out() - output function
            f_expect() - expected output function
            f_deriv_loss = d(f_loss)/d(f_out)

        (*) Example of loss function: Euclidean distance

                f_loss(y_out) = (1/2)*((y_out - y_expect)^2)
                d(f_loss)/d(x) = x - y_expect
                f_deriv_loss(x) = d(f_loss)/d(x)

    (*) A neuron (also known as a "node") processes an input, represented
        by a vector of numbers, into an output signal by means of "weights"
        and an "activation function".

                   w_1
            in_1 ---*---+
                        |
                   w_2  |
            in_2 ---*---+
                        |    |-------\       |-----------\
                        +--->|  Sum   |---> |  f_activ()  |---> output
                        |    |-------/       |-----------/
               ...      |
                        |
                   w_n  |
            in_n ---*---+

        For a neuron with multiple inputs:

            f_preact() = Sum[k](w_k * in_k)
            f_out() = f_activ(f_preact())
            f_deriv_act() = d(f_activ(z))/d(z)

            where:
                in_k        - is the input k
                w_k         - is the weight of input k
                f_preact    - is the preactivation of the neuron
                f_out       - is the output of the neuron

            (*) Examples of activation functions:

                (*) sigmoid

                sigmoid(x) = 1/(1 + e^(-x))

                f_activ(z) = sigmoid(z)
                f_deriv_act(z) = (d/d_z)(sigmoid)(z)
                               = sigmoid(z)(1 - sigmoid(z))
                               = f_activ(z)(1 - f_activ(z))

                (*) tanh

                tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))

                f_activ(z) = tanh(z)
                f_deriv_act(z) = (d/d_z)(tanh)(z)
                               = 1 - (tanh(z))^2
                               = 1 - (f_activ(z))^2

            f_out(w, in) = f_activ(f_preact(w, in))

    (*) The topology of a multi-layer perceptron (MLP) consists of a
        sequence of neuron layers. The last layer (L), also referred to as the
        "output layer", provides the predictive output activation that is
        compared to the expected output. The difference between the predictive
        and the expected output represents the residual.

        A simple example of an MLP topology is provided in the following
        diagram.

             IN_A --->----- (C) --->-----\
                 \          /  \          (E) ->- \
                  \   /->--/    \   /->--/         \
                   \ /           \ /                \
                    \             \                  (G) -> OUTPUT
                   / \           / \                /
                  /   \->--\    /   \->--\         /
                 /          \  /          (F) ->- /
             IN_B --->----- (D) --->-----/

        Here, the output layer, or the last layer, consists of
        one neuron: G. It is preceded by a hidden layer consisting of neurons
        E and F. This is preceded by another layer formed by neurons C and D.
        Layer "L" (last) consists of neuron "G."

        In order to simplify, the present explanation will consider a
        simple topology with only one neuron per layer.

            IN ->- ... ->- (p) ->- (q) ->- (r) ->- ... ->- (L) -> OUTPUT

        The L node is the last node providing the output.

        In order to compute the gradient of each node's weight, a helper
        function is computed for each node/neuron. This is referred to
        as "delta," or alternatively, the "error term" of a neuron.

            delta() = d(f_loss)/d(f_preact)

        It represents the variation of the loss function with respect to
        the preactivation of the neuron.

            f_loss() - loss function of the last layer neurons, a function
                        of the neuron output

    (*) The delta function on a generic layer neuron (q):

        delta_q() = d(f_loss)/d(f_preact_q)

        (*) layer (q) is followed by layer (r) and preceded by (p):

        delta_q = (d(f_loss)/d(f_preact_r))*(d(f_preact_r)/d(f_preact_q))

            delta_r = d(f_loss)/d(f_preact_r)

        delta_q = (delta_r(f_preact_r))*(d(f_preact_r)/d(f_preact_q))

            d(f_preact_r)/d(f_preact_q) =
                    = (d(f_preact_r)/d(f_out_q))*(d(f_out_q)/d(f_preact_q))

                f_preact_r = w_r * in_r

                    (*) note that input to r is the output from q
                        in_r = f_out_q

                f_preact_r = w_r * f_out_q

                d(f_preact_r)/d(f_out_q) = d(w_r * f_out_q)/d(f_out_q)
                d(f_preact_r)/d(f_out_q) = w_r

            d(f_preact_r)/d(f_preact_q) = (w_r)*(d(f_out_q)/d(f_preact_q))

                (*) notation:
                    f_deriv_act = d(f_out)/d(f_preact)

            d(f_preact_r)/d(f_preact_q) = (w_r)*(f_deriv_act(f_preact_q))

        delta_q = delta_r * w_r * f_deriv_act
        delta_q(f_preact_q) = delta_r(f_preact_r) * w_r * f_deriv_act(f_preact_q)

    (*) The delta function on the last layer neuron:

        delta_L() = d(f_loss)/d(f_preact_L)
            = (d(f_loss)/d(f_out_L))*(d(f_out_L)/d(f_preact_L))
            = (d(f_loss)/d(f_out_L))*(f_deriv_act(f_preact_L))

                d(f_loss)/d(f_out) = f_deriv_loss(f_out)

        delta_L() = d(f_loss)/d(f_preact_L)
            = (d(f_loss)/d(f_out_L))*(f_deriv_act(f_preact_L))
            = (f_deriv_loss(f_out))*(f_deriv_act(f_preact_L))

        delta_L() = (f_deriv_loss(f_out()))*(f_deriv_act(f_preact_L))

    (*) The gradient of the loss function can be expressed as:

        d(f_loss)/d(w_q) = (d(f_loss)/d(f_preact_q))*(d(f_preact_q)/d(w_q))
        d(f_loss)/d(w_q) = (delta_q(f_preact_q))*(d(f_preact_q)/d(w_q))

            f_preact_q(w_q, in_q) = w_q * in_q
            d(f_preact_q)/d(w_q) = in_q

        d(f_loss)/d(w_q) = delta_q(f_preact_q) * in_q

    (*) Using the previous expressions, the gradient descent update of the
        weights can be computed as:

        w_k_new = w_k_crt - learn_rate * (d(f_loss)/d(w_k))(w_k_crt)

    (*) Note that the effect of the neurons' biases has not been detailed,
        but they do operate as regular weights with a constant input of one.

'''


# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import math
import random

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
random.seed(1001)

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class MlpUnivPredict :

    def __init__(self, aux_param = None) :
        MlpWrk.Init(self, aux_param)

    version = 0.901

    def __repr__(self):
        '''Returns representation of the object'''
        return("{}({!r})".format(self.__class__.__name__, self.aux_param))

    def fit(self, X, y) :
        # Initialize the layers
        MlpWrk.WorkFit(self, X, y)

    def predict(self, X) :
        ret_data = MlpWrk.WorkPredict(self, X)
        return ret_data

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class MlpWrk :

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def Init(object, aux_param) :

        if aux_param == None :
            object.aux_param = {}
        else :
            object.aux_param = aux_param

        if not 'trace_cfg' in object.aux_param :
            object.aux_param['trace_cfg'] = {'trace_enable': True}

        MlpWrk.Misc.trace_enable = object.aux_param['trace_cfg']['trace_enable']
        MlpWrk.Misc.trace_epoch_print = MlpWrk.Misc.trace_enable
        if not 'epoch_period' in object.aux_param['trace_cfg'] :
            MlpWrk.Misc.trace_epoch_period = 10
        else :
            MlpWrk.Misc.trace_epoch_period = object.aux_param['trace_cfg'][
                                                       'epoch_period']

        MlpWrk.Misc.TracePrint(">> trace - aux_param:", aux_param)

        if not 'learn_rate' in object.aux_param :
            object.aux_param['learn_rate'] = 0.1
        if not 'no_epochs' in object.aux_param :
            object.aux_param['no_epochs'] = 500
        if not 'hid_dim_lst' in object.aux_param :
            object.aux_param['hid_dim_lst'] = [6, 4]
        if not 'activation' in object.aux_param :
            object.aux_param['activation'] = 'sigmoid'
        if not 'loss' in object.aux_param :
            object.aux_param['loss'] = 'mse'
        if not 'pattern_init' in object.aux_param :
            object.aux_param['pattern_init'] = False
        if not 'predict_mode' in object.aux_param :
            object.aux_param['predict_mode'] = 'auto'
        if not 'onehot_limit' in object.aux_param :
            object.aux_param['onehot_limit'] = 10
        if not 'scale_input' in object.aux_param :
            object.aux_param['scale_input'] = True

        object.neural_net = None

        if object.aux_param['activation'] == 'sigmoid' :
            object.fn_activation = MlpWrk.Misc.FnSigmoidActivation
            object.fn_deriv_activ = MlpWrk.Misc.FnSigmoidDerivActiv
            object.fn_activ_min = 0
            object.fn_activ_max = 1
        elif object.aux_param['activation'] == 'tanh' :
            object.fn_activation = MlpWrk.Misc.FnTanhActivation
            object.fn_deriv_activ = MlpWrk.Misc.FnTanhDerivActiv
            object.fn_activ_min = -1
            object.fn_activ_max = 1

        if object.aux_param['loss'] == 'mse' :
            object.fn_loss = MlpWrk.Misc.FnMeanSquaredErrorLoss
            object.fn_deriv_loss = MlpWrk.Misc.FnMeanSquaredErrorDerivLoss
        elif object.aux_param['loss'] == 'mae' :
            object.fn_loss = MlpWrk.Misc.FnMeanAbsoluteErrorLoss
            object.fn_deriv_loss = MlpWrk.Misc.FnMeanAbsoluteErrorDerivLoss
        elif object.aux_param['loss'] == 'pow3' :
            object.fn_loss = MlpWrk.Misc.FnMeanPower3ErrorLoss
            object.fn_deriv_loss = MlpWrk.Misc.FnMeanPower3ErrorDerivLoss
        elif object.aux_param['loss'] == 'pow4' :
            object.fn_loss = MlpWrk.Misc.FnMeanPower4ErrorLoss
            object.fn_deriv_loss = MlpWrk.Misc.FnMeanPower4ErrorDerivLoss

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def WorkFit(object, data_X, data_y) :

        MlpWrk.Misc.TracePrint(">> trace - WorkFit, data_X:", data_X)
        MlpWrk.Misc.TracePrint(">> trace - WorkFit, data_y:", data_y)

        # Process attributes
        onehot_max_col = object.aux_param['onehot_limit']
        input_col_no = len(data_X[0])
        scale_input = object.aux_param['scale_input']
        ret_data = Util.PreprocessMixToNum(data_X, onehot_max_col, scale_input)
        x_attr, insert_list, attr_dict_lst, num_param_lst = ret_data
        expand_col_no = len(x_attr[0])

        MlpWrk.Misc.TracePrint(">> trace - WorkFit, x_attr:", x_attr)
        MlpWrk.Misc.TracePrint(">> trace - WorkFit, attr_dict_lst:", attr_dict_lst)
        MlpWrk.Misc.TracePrint(">> trace - WorkFit, num_param_lst:", num_param_lst)

        object.attr_dict_lst = attr_dict_lst
        object.num_param_lst = num_param_lst
        predict_mode = object.aux_param['predict_mode']

        # Process output
        ret_data = MlpWrk.PreprocessOutput(object, data_y)
        expect_output = ret_data

        object.no_epochs = object.aux_param['no_epochs']
        object.learn_rate = object.aux_param['learn_rate']
        object.hid_dim_lst = object.aux_param['hid_dim_lst']

        input_dim = len(x_attr[0])
        object.input_no = input_dim
        object.layer_lst = None
        object.input_lst = None
        object.tmp_count = 1

        ret_val = MlpWrk.NeuNet.Train(object, x_attr, expect_output)
        return ret_val

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def WorkPredict(object, data_X) :

        # translate attribute to one_hot version
        attr_dict_lst = object.attr_dict_lst
        num_param_lst = object.num_param_lst
        scale_input = object.aux_param['scale_input']
        x_attr = Util.ConvertMixToNum(data_X, attr_dict_lst, num_param_lst, scale_input)
        output_list = MlpWrk.NeuNet.Predict(object, x_attr)
        output_no = len(output_list)
        predict_idx_lst = []
        predict_output = []

        if object.regress_mode :
            # regression
            # descale output
            adj_out_list = []
            for elem in output_list :
                new = MlpWrk.Misc.FnDescaleValue(elem[0], object.y_scale, object.y_offset,
                                        object.activ_scale, object.activ_offset)
                adj_out_list.append(new)
            predict_output = adj_out_list[:]
        else :
            # classification
            for idx_out in range(output_no) :
                crt_output = output_list[idx_out]
                max_idx = crt_output.index(max(crt_output))
                predict_idx_lst.append(max_idx)

            # translate indexes to outputs
            for crt_idx in predict_idx_lst :
                select_output = object.distinct_list[crt_idx][0]
                predict_output.append(select_output)

        return predict_output

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def PreprocessOutput(object, in_y) :
        # import sys
        data_y = Util.ListDataConvert(in_y)
        if not data_y == [] :
            if (isinstance(data_y[0], list)) :
                if len(data_y[0]) == 1 :
                    # must be a one column matrix
                    transpose_data = Util.MatrixTranspose(data_y)
                    data_y = transpose_data[0]
                else :
                    data_y = None
        int_is_num = True
        if object.aux_param['predict_mode'] == 'auto' :
            detect_regression = Util.RegressDetect(data_y, int_is_num)
            MlpWrk.Misc.TracePrint(">> trace - detect_regression:", detect_regression)
            select_regression = detect_regression
        elif object.aux_param['predict_mode'] == 'classif' :
            select_regression = False
        else :
            # if object.aux_param['predict_mode'] == 'regress'
            select_regression = True

        if select_regression :
            predictive_mode = 'regression'
        else :
            predictive_mode = 'classification'
        # MlpWrk.Misc.TracePrint(">> trace - select_regression:", select_regression)
        MlpWrk.Misc.TracePrint(">> trace - predictive_mode:", predictive_mode)

        # scale output
        out_min = object.fn_activ_min
        out_max = object.fn_activ_max
        out_delta = out_max - out_min

        if select_regression :
            activ_scale = 0.5 * out_delta
            activ_offset = out_min + 0.25 * out_delta

            data_min = min(data_y)
            data_max = max(data_y)
            data_offset = data_min
            data_scale = (data_max - data_min)
            if data_scale == 0 :
                data_scale = 1
        else :
            activ_scale = 1.0 * out_delta
            activ_offset = out_min
            data_scale = 1
            data_offset = 0

        if not select_regression :
            out_dict = Util.GetDistinctElements(data_y)
            distinct_list = list(out_dict.items())
            object.distinct_list = distinct_list
            output_dim = len(distinct_list)
        else :
            output_dim = 1

        object.y_offset = data_offset
        object.y_scale = data_scale
        object.activ_offset = activ_offset
        object.activ_scale = activ_scale

        y_expect = data_y
        regress_flag = select_regression
        MlpWrk.Misc.TracePrint(">> trace - WorkFit, y_expect:", y_expect)

        input_rows = len(in_y)
        if object.aux_param['predict_mode'] == 'classif' :
            object.regress_mode = False
        elif object.aux_param['predict_mode'] == 'regress' :
            object.regress_mode = True
        else :
            # if object.aux_param['predict_mode'] == 'auto'
            object.regress_mode = regress_flag

        object.output_no = output_dim

        if not object.regress_mode :
            # classification mode - discretize expected output
            scaled_true = MlpWrk.Misc.FnScaleValue(1.0, data_scale,
                                        data_offset, activ_scale, activ_offset)
            scaled_false = MlpWrk.Misc.FnScaleValue(0.0, data_scale,
                                        data_offset, activ_scale, activ_offset)
            expect_onehot = []
            for idx_inrow in range(input_rows) :
                # Train
                crt_output = y_expect[idx_inrow]
                crt_onehot_out = []
                for crt_idx in range(object.output_no) :
                    crt_pair = object.distinct_list[crt_idx]
                    if crt_pair[0] == crt_output :
                        # crt_onehot_out.append(1.0)
                        crt_onehot_out.append(scaled_true)
                    else :
                        # crt_onehot_out.append(0.0)
                        crt_onehot_out.append(scaled_false)
                expect_onehot.append(crt_onehot_out)
            expect_output = expect_onehot
        else :
            expect_regress = []
            for idx_inrow in range(input_rows) :
                # Train
                crt_output = y_expect[idx_inrow] * 1.0
                scaled_out = MlpWrk.Misc.FnScaleValue(crt_output, data_scale,
                                            data_offset, activ_scale, activ_offset)
                # expect_regress.append([crt_output])
                expect_regress.append([scaled_out])
            expect_output = expect_regress

        ret_val =expect_output
        return ret_val

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    class NeuNet :

        # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        class Neuron :
            # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            @staticmethod
            def Init(object, weights, bias, fn_activation) :
                object['weights'] = weights
                object['bias'] = bias
                object['fn_activation'] = fn_activation
                object['last_preact'] = 0
                object['last_activation'] = 0
                object['delta'] = 0

            # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            @staticmethod
            def CalcOutput(object, stimuli) :
                fn_activation = object['fn_activation']
                # <theory-ref>
                #   f_out() = f_activ(f_preact())
                z = MlpWrk.Misc.FnPreact(stimuli, object['weights'], object['bias'])
                object['last_preact'] = z
                out = fn_activation(z)
                object['last_activation'] = out
                return out

            # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        @staticmethod
        def Train(object, x_attr, y_expect) :
            # Initialisation

            fn_activation = object.fn_activation
            fn_loss = object.fn_loss
            pat_init_flag = object.aux_param['pattern_init']
            # Initialize the layers
            input_rows = len(x_attr)
            layer_network = []
            prev_elem_no = object.input_no
            total_dim_lst = object.hid_dim_lst[:]
            # append output layer dimension
            total_dim_lst.append(object.output_no)
            # generate neural layout
            for crt_layer_dim in total_dim_lst :
                crt_layer_list = []
                for crt_idx_layer in range(crt_layer_dim) :
                    weight_list = []
                    for crt_idx_prev in range(prev_elem_no) :
                        if not pat_init_flag :
                            w_value = random.random()
                        else :
                            w_value = (object.tmp_count % 2) * 1
                        weight_list.append(w_value)
                        object.tmp_count += 1
                    if not pat_init_flag :
                        w_value = random.random()
                    else :
                        w_value = 0.5
                    crt_neur = {}
                    MlpWrk.NeuNet.Neuron.Init(crt_neur, weight_list, w_value, fn_activation)
                    crt_layer_list.append(crt_neur)
                prev_elem_no = crt_layer_dim
                layer_network.append(crt_layer_list)
            object.layer_net = layer_network

            # Training
            MlpWrk.Misc.TracePrint(">> trace - after initialization")

            MlpWrk.Misc.PrintNetState(object.layer_net)
            epochs_no = object.aux_param['no_epochs']
            trace_epoch_period = MlpWrk.Misc.trace_epoch_period
            shfl_idx_lst = list(range(input_rows))
            for idx_epoch in range(epochs_no) :
                trace_epoch_idx = idx_epoch
                if MlpWrk.Misc.trace_enable :
                    if (trace_epoch_idx == epochs_no - 1
                        or trace_epoch_idx % trace_epoch_period == 0) :
                        MlpWrk.Misc.trace_epoch_print = True
                    else :
                        MlpWrk.Misc.trace_epoch_print = False
                sum_error = 0
                # random.shuffle(shfl_idx_lst)
                for idx_inrow in shfl_idx_lst :
                    # Train
                    crt_input_attr = x_attr[idx_inrow]
                    crt_expect_out = y_expect[idx_inrow]
                    crt_output = MlpWrk.NeuNet.ForwardPropagation(object, crt_input_attr)
                    MlpWrk.NeuNet.BackwardPropagation(object, crt_expect_out)
                    MlpWrk.Misc.TracePrint(">> trace - after backprop")
                    MlpWrk.Misc.PrintNetState(object.layer_net)
                    for idx_outp in range(object.output_no) :
                        out = crt_output[idx_outp]
                        expect = crt_expect_out[idx_outp]
                        residual = out - expect
                        loss_err = fn_loss(residual)
                        sum_error += loss_err
                MlpWrk.Misc.TracePrint(">> epoch: %d, error: %.9f" % (trace_epoch_idx, sum_error))
                MlpWrk.Misc.TracePrint(">>")

        # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        @staticmethod
        def Predict(object, x_attr) :
            input_rows = len(x_attr)
            input_dim = len(x_attr[0])
            if input_dim != object.input_no :
                return None
            predict_out_lst = []
            for idx_inrow in range(input_rows) :
                crt_input_attr = x_attr[idx_inrow]
                crt_output = MlpWrk.NeuNet.ForwardPropagation(object, crt_input_attr)
                predict_out_lst.append(crt_output)
            return predict_out_lst

        # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        @staticmethod
        def ForwardPropagation(object, input_stimuli) :
            fn_activation = object.fn_activation

            # Compute the outputs of the network
            object.input_lst = input_stimuli
            crt_stimuli = input_stimuli
            for crt_layer in object.layer_net :
                MlpWrk.Misc.TracePrint(">> trace - fw propag - crt_stimuli:", crt_stimuli)
                next_stimuli = []
                for crt_neur in crt_layer :
                    crt_f_out = MlpWrk.NeuNet.Neuron.CalcOutput(crt_neur, crt_stimuli)
                    next_stimuli.append(crt_f_out)
                crt_stimuli = next_stimuli
            network_output = crt_stimuli
            MlpWrk.Misc.TracePrint(">> trace - fw propag - output:", network_output)
            return network_output

        # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        @staticmethod
        def BackwardPropagation(object, output_expect) :
            # update deltas
            layer_no = len(object.layer_net)
            last_layer = object.layer_net[-1]
            fn_activation = object.fn_activation
            fn_deriv_activ = object.fn_deriv_activ
            fn_loss = object.fn_loss
            fn_deriv_loss = object.fn_deriv_loss

            # first update output (last) layer
            for crt_idx_neur in range(object.output_no) :
                crt_neur = last_layer[crt_idx_neur]
                crt_out = crt_neur['last_activation']
                crt_exp = output_expect[crt_idx_neur]
                residual = crt_out - crt_exp
                f_deriv_loss = fn_deriv_loss(residual)
                f_deriv_act = fn_deriv_activ(crt_neur['last_preact'])
                # <theory-ref>
                #   delta_L() = (f_deriv_loss(f_res()))*(f_deriv_act(f_preact_L))
                delta_last = f_deriv_loss * f_deriv_act
                crt_neur['delta'] = delta_last

            # update remaining layers
            remain_layers = object.layer_net[:-1]
            prev_layer = last_layer

            for crt_idx_layer in reversed(range(layer_no - 1)) :
                crt_layer = remain_layers[crt_idx_layer]
                layer_error_list = []
                for crt_idx_neur in range(len(crt_layer)) :
                    crt_neur = crt_layer[crt_idx_neur]
                    err_delta = 0
                    for prev_idx_neur in range(len(prev_layer)) :
                        prev_neur = prev_layer[prev_idx_neur]
                        prev_weight = prev_neur['weights'][crt_idx_neur]
                        prev_delta = prev_neur['delta']
                        # <theory-ref>
                        #   d(f_preact_r)/d(f_preact_q) = (w_r)*(d(f_out_q)/d(f_preact_q))
                        crt_term = prev_weight * prev_delta
                        err_delta += crt_term
                    neur_loss_err = fn_loss(err_delta)
                    layer_error_list.append(neur_loss_err)
                    delta_residual = err_delta
                    f_deriv_act = fn_deriv_activ(crt_neur['last_preact'])
                    # <theory-ref>
                    #   delta_q(f_preact_q) = delta_r(f_preact_r) * w_r * f_deriv_act(f_preact_q)
                    new_delta = delta_residual * f_deriv_act
                    crt_neur['delta'] = new_delta
                prev_layer = crt_layer
                layer_error = sum(layer_error_list)
                MlpWrk.Misc.TracePrint(">>   idx_layer: %4d, layer_error: %.9f" % (crt_idx_layer, layer_error))

            # update weights
            crt_stimuli = object.input_lst
            for crt_idx_layer in range(layer_no) :
                crt_layer = object.layer_net[crt_idx_layer]
                next_stimuli = []
                for crt_idx_neur in range(len(crt_layer)) :
                    crt_neur = crt_layer[crt_idx_neur]
                    for crt_idx_inp in range(len(crt_stimuli)) :
                        crt_weight = crt_neur['weights'][crt_idx_inp]
                        # <theory-ref>
                        #   w_k_new = w_k_crt - learn_rate * (d(f_loss)/d(w_k))(w_k_crt)
                        dfloss_dw = crt_neur['delta'] * crt_stimuli[crt_idx_inp]
                        crt_diff = object.learn_rate * dfloss_dw
                        new_weight = crt_weight - crt_diff
                        crt_neur['weights'][crt_idx_inp] = new_weight
                    crt_weight = crt_neur['bias']
                    crt_diff = object.learn_rate * crt_neur['delta']
                    new_weight = crt_weight - crt_diff
                    crt_neur['bias'] = new_weight
                    crt_neur_out = fn_activation(crt_neur['last_preact'])
                    next_stimuli.append(crt_neur_out)
                crt_stimuli = next_stimuli
        # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    class Misc :

        trace_enable = False
        trace_epoch_print = False

        # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        @staticmethod
        def FnPreact(stimuli, weights, bias) :
            # <theory-ref>
            #   f_preact() = Sum[k](w_k * in_k)
            z = Util.FnDotProd(stimuli, weights) + bias
            return z

        # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        @staticmethod
        def FnSigmoidActivation(in_x) :
            # sigmoid
            return 1 / (1.0 + math.exp(-in_x))

        # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        @staticmethod
        def FnSigmoidDerivActiv(in_x) :
            # sigmoid derivative
            return MlpWrk.Misc.FnSigmoidActivation(in_x) * (1 - MlpWrk.Misc.FnSigmoidActivation(in_x))

        # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        @staticmethod
        def FnTanhActivation(in_x) :
            # tanh
            exp_pos_2x = math.exp(2*in_x)
            return (exp_pos_2x - 1) / (exp_pos_2x + 1)

        # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        @staticmethod
        def FnTanhDerivActiv(in_x) :
            # tanh derivative
            return (1 - (MlpWrk.Misc.FnTanhActivation(in_x))**2)

        # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        @staticmethod
        def FnMeanSquaredErrorLoss(residual) :
            # residual = output - expect
            loss = (1/2)*(residual**2)
            return loss

        # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        @staticmethod
        def FnMeanSquaredErrorDerivLoss(residual) :
            return residual

        # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        @staticmethod
        def FnMeanAbsoluteErrorLoss(residual) :
            # residual = output - expect
            loss = abs(residual)
            return loss

        # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        @staticmethod
        def FnMeanAbsoluteErrorDerivLoss(residual) :
            # residual = output - expect
            if residual > 0 :
                deriv = 1
            elif residual < 0 :
                deriv = -1
            else :
                deriv = 0
            return deriv

        # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        @staticmethod
        def FnMeanPower3ErrorLoss(residual) :
            # residual = output - expect
            loss = abs((1/3)*(residual**3))
            return loss

        # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        @staticmethod
        def FnMeanPower3ErrorDerivLoss(residual) :
            # residual = output - expect
            if residual > 0 :
                deriv = (residual**2)
            elif residual < 0 :
                deriv = -(residual**2)
            else :
                deriv = 0
            return deriv

        # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        @staticmethod
        def FnMeanPower4ErrorLoss(residual) :
            # residual = output - expect
            loss = (1/4)*(residual**4)
            return loss

        # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        @staticmethod
        def FnMeanPower4ErrorDerivLoss(residual) :
            # residual = output - expect
            deriv = (residual**3)
            return deriv

        # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        @staticmethod
        def FnScaleValue(in_val, orig_scale, orig_offset, activ_scale, activ_offset) :
            out_val = ((in_val - orig_offset)/orig_scale)*activ_scale + activ_offset
            return out_val

        # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        @staticmethod
        def FnDescaleValue(in_val, orig_scale, orig_offset, activ_scale, activ_offset) :
            out_val = (((in_val - activ_offset)*orig_scale)/activ_scale + orig_offset)
            return out_val

        # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        @staticmethod
        def PrintNetState(in_network) :
            MlpWrk.Misc.TracePrint(">>    net_state")
            for idx_layer in range(len(in_network)):
                MlpWrk.Misc.TracePrint(">>    net_state - idx_layer:", idx_layer)
                crt_layer = in_network[idx_layer]
                for idx_neur in range(len(crt_layer)):
                    MlpWrk.Misc.TracePrint(">>    net_state - idx_neur:", idx_neur)
                    crt_neuron = crt_layer[idx_neur]
                    neur_weigths = []
                    for idx_weight in range(len(crt_neuron['weights'])):
                        crt_weight = crt_neuron['weights'][idx_weight]
                        neur_weigths.append(crt_weight)
                    neur_weigths.append(crt_neuron['bias'])
                    MlpWrk.Misc.TracePrint(">>    net_state - neur_weigths:", neur_weigths)
                    MlpWrk.Misc.TracePrint(">>    net_state - delta:", crt_neuron['delta'])
            MlpWrk.Misc.TracePrint(">>")

        # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        @staticmethod
        def TracePrint(*args):
            if MlpWrk.Misc.trace_enable :
                if MlpWrk.Misc.trace_epoch_print :
                    print(*args)

        # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class Util :

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def CrtTimeStamp(display_flag = True) :
        import datetime

        in_time_stamp = datetime.datetime.now()
        time_str = in_time_stamp.strftime("%Y-%m-%d %H:%M:%S")
        out_str = "time_stamp: %s" % (time_str)
        if display_flag :
            print(out_str)
        return out_str

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def SepLine(display_flag = True) :
        separator_string = ">-" + 78*"-"
        print(separator_string)

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def UpdateDict(target_dict, source_dict) :
        for key, value in source_dict.items():
            target_dict[key] = value

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def GetDistinctElements(lst) :
        element_dict = {}
        for element in lst :
            if element in element_dict :
                element_dict[element] += 1
            else :
                element_dict[element] = 1
        return element_dict

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def FnDotProd(vect_1, vect_2) :
        # dot product
        crt_scalar = 0
        for el_1, el_2 in zip(vect_1, vect_2) :
            crt_prod = el_1 * el_2
            crt_scalar += crt_prod
        return crt_scalar

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def ListDataConvert(in_data) :
        import numpy as np
        import pandas as pd

        if (isinstance(in_data, list)) :
            # check whether is a one column array
            lst_data = in_data.copy()
        elif (isinstance(in_data, np.ndarray)) :
            lst_data = in_data.tolist()
        elif (isinstance(in_data, pd.core.arrays.PandasArray)) :
            lst_data = in_data.tolist()
        elif (isinstance(in_data, pd.core.frame.DataFrame)) :
            lst_data = in_data.values.tolist()
        elif (isinstance(in_data, pd.core.series.Series)) :
            lst_data = in_data.values.tolist()
        else :
            lst_data = None

        return lst_data

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def GetCol( in_array, in_col ) :
        ret_item = []
        if not isinstance(in_array, list) :
            ret_item = None
        else :
            row_no = len(in_array)
            if not isinstance(in_array[0], list) :
                ret_item = None
            else :
                ret_item = []
                for crt_idx_row in range(row_no) :
                    crt_row_len = len(in_array[crt_idx_row])
                    if in_col < crt_row_len :
                        ret_item.append((in_array[crt_idx_row][in_col]))
                    else :
                        ret_item.append(None)
        return(ret_item)

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def MatrixTranspose( in_array ) :
        if in_array == [] : return []
        if not isinstance(in_array[0], list) :
            transp_data = list(in_array)
        else :
            transp_data = []
            col_no = len(in_array[0])
            for crt_idx_col in range(col_no) :
                crt_vect = Util.GetCol( in_array, crt_idx_col )
                transp_data.append(crt_vect)
        ret_item = transp_data
        return(ret_item)

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def OrderedFreqCount(in_symbol_sequence_list):
        """
        Returns a list that contains info about the frequency of
        items in parameter input list (in_symbol_sequence_list).
        The list is sorted on no of occurences order

        Params:
            in_symbol_sequence_list
                sequence of symbol occurences

        returns:
            out_list
                The output list has the following structure:
                each row (first level list)  has:
                    first column the element itself from the list
                    second column the no of occurences
                    third column the list of indexes containing the element
        """
        from operator import itemgetter

        in_len = len(in_symbol_sequence_list)
        idx = 0
        out_list = []
        for in_el in in_symbol_sequence_list:
            found_match = 0
            for out_el in out_list:
                if in_el == out_el[0]:
                    out_el[1] = out_el[1] + 1
                    out_el[2].append(idx)
                    found_match = 1
                    break
            if found_match == 0:
                out_list.append([in_el, 1, []])
                # append index into third column
                out_list[-1][2].append(idx)
            idx = idx + 1

        out_list.sort(key=itemgetter(1), reverse=True)
        return out_list

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def SummaryFreqCount(in_symbol_sequence_list):
        """
        Returns a summary about the frequency of
        items in parameter input list (in_symbol_sequence_list).
        The list is sorted on no of occurences order

        Params:
            in_symbol_sequence_list
                sequence of symbol occurences

        returns:
            ret_no_of_distinct_elems
            ret_elem_list
                List of distinct elements
            ret_count_list
                List with counts of each element matching ret_elem_list
        """
        count_data = Util.OrderedFreqCount(in_symbol_sequence_list)
        distinct_elem_no, elem_list, count_list = Util.CountDataToFreqLists(count_data)

        return distinct_elem_no, elem_list, count_list

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def CountDataToFreqLists(in_freq_count_data):
        """
        Returns a summary of info about the frequency of
        items in parameter input list (in_freq_count_data).
        The lists are sorted on no of occurences order

        Params:
            in_freq_count_data
                sequence of symbol occurences

        returns:
            ret_no_of_distinct_elems
            ret_elem_list
                List of distinct elements
            ret_count_list
                List with counts of each element matching ret_elem_list
        """
        # determine no of distinct elements
        distinct_elem_no = len(in_freq_count_data)

        # Filter out the index lists
        elem_count_pairs = [ elem[:2] for elem in in_freq_count_data ]

        elem_list = [ x[0] for x in elem_count_pairs ]
        count_list = [ x[1] for x in elem_count_pairs ]

        return distinct_elem_no, elem_list, count_list

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def SelectiveOneHotProc(in_vector, in_max_col = 0) :

        fn_ret_data = []
        # one iteration loop to allow unified return through loop breaks
        for dummy_idx in range(1) :
            vect_len = len(in_vector)
            ret_tuple = Util.SummaryFreqCount(in_vector)
            crt_types_no, crt_id_list, crt_count_list = ret_tuple
            if in_max_col == 0 :
                label_no = crt_types_no - 1
            else :
                label_no = min(crt_types_no - 1, in_max_col)
            one_hot_dict = {}
            for crt_label_idx in range(label_no) :
                crt_label_id = crt_id_list[crt_label_idx]
                one_hot_dict[crt_label_id] = crt_label_idx
                # one_hot_dict[crt_label_id] = (label_no - 1) - crt_label_idx
            if label_no == 0 :
                one_hot_list = [[0] * vect_len]
            else :
                one_hot_list = []
                for crt_label_idx in range(label_no) :
                    one_hot_list.append([])
                for crt_idx in range(vect_len) :
                    crt_elem = in_vector[crt_idx]
                    for crt_label_idx in range(label_no) :
                        crt_label_id = crt_id_list[crt_label_idx]
                        if crt_elem == crt_label_id :
                            append_elem = 1
                        else :
                            append_elem = 0
                        one_hot_list[crt_label_idx].append(append_elem)
            fn_ret_data = one_hot_list, one_hot_dict
        return fn_ret_data

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def ValidateFloat(in_val) :
        import numpy as np
        if not isinstance(in_val, float) :
            fn_ret_status = False
            fn_ret_translate = in_val
        elif np.isnan(in_val) :
            fn_ret_status = False
            fn_ret_translate = "nan"
        elif in_val == float('inf') :
            fn_ret_status = False
            fn_ret_translate = "+inf"
        elif in_val == float('-inf') :
            fn_ret_status = False
            fn_ret_translate = "-inf"
        else :
            fn_ret_status = True
            fn_ret_translate = in_val
        ret_tuple = fn_ret_status, fn_ret_translate
        return ret_tuple

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def NumericalCheck( in_value, int_is_num_flag = True ) :
        """
            Check if numerical. If non regular float, result is false and
            translated value returned
        """
        if not int_is_num_flag :
            if isinstance(in_value, float) :
                float_flag, valid_val = Util.ValidateFloat(in_value)
                fn_ret = (float_flag, valid_val)
            else :
                fn_ret = (False, in_value)
        else :
            if isinstance(in_value, float) :
                float_flag, valid_val = Util.ValidateFloat(in_value)
                fn_ret = (float_flag, valid_val)
            elif isinstance(in_value, int) :
                fn_ret = (True, in_value)
            else :
                fn_ret = (False, in_value)
        return fn_ret

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def AnalyzeVectorType(in_vector, in_num_precalc, in_num_scale_flag) :

        import statistics

        fn_ret_data = (False, False, [])
        # one iteration loop to allow unified return through loop breaks
        for dummy_idx in range(1) :
            MlpWrk.Misc.TracePrint(">> trace - AnalyzeVectorType, in_vector:", in_vector)
            opmode_intisnum = True
            has_num_flag = False
            has_categ_flag = False
            has_none_flag = False
            num_lst_svg = None
            num_lst_min = None
            num_lst_max = None
            if not in_num_precalc == None :
                num_lst_svg = in_num_precalc['avg']
                num_lst_min = in_num_precalc['min']
                num_lst_max = in_num_precalc['max']
            vect_len = len(in_vector)
            num_list = []
            for crt_idx in range(vect_len) :
                crt_elem = in_vector[crt_idx]
                is_numerical, translate_value = Util.NumericalCheck(crt_elem, opmode_intisnum)
                if is_numerical :
                    num_list.append(crt_elem)
                    has_num_flag = True
                elif crt_elem == None :
                    has_none_flag = True
                else :
                    has_categ_flag = True
            is_numerical = not has_categ_flag
            if is_numerical :
                # numerical, but could contain None elements (missing placeholders)
                if in_num_precalc == None :
                    num_lst_svg = statistics.mean(num_list)
                    num_lst_min = min(num_list)
                    num_lst_max = max(num_list)
                reiterate_flag = False
                if has_none_flag :
                    reiterate_flag = True
                if in_num_scale_flag :
                    reiterate_flag = True
                if reiterate_flag :
                    convert_list = []
                    for crt_idx in range(vect_len) :
                        crt_elem = in_vector[crt_idx]
                        if crt_elem == None :
                            crt_elem = num_lst_svg
                        if in_num_scale_flag :
                            # scale value
                            orig_scale = num_lst_max - num_lst_min
                            orig_offset = num_lst_min
                            target_scale = 1
                            target_offset = 0
                            val_scale = MlpWrk.Misc.FnScaleValue(crt_elem, orig_scale,
                                                    orig_offset, target_scale, target_offset)
                            convert_list.append(val_scale)
                        else :
                            convert_list.append(crt_elem)
                    out_list = convert_list
                else :
                    out_list = in_vector[:]
            else :
                # Categorical or mixed
                out_list = in_vector[:]
            fn_ret_data = (is_numerical, has_none_flag,
                                out_list, num_lst_svg, num_lst_min, num_lst_max)
            MlpWrk.Misc.TracePrint(">> trace - AnalyzeVectorType, fn_ret_data:", fn_ret_data)
        return fn_ret_data

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def RegressDetect( in_vect, in_int_is_num = True) :
        """
            Parse vector and check if vector has valid regress numerical elements.

        """
        regress_flag = False
        num_list = []
        for crt_elem in in_vect :
            is_numerical, translate_value = Util.NumericalCheck(crt_elem, in_int_is_num)
            if is_numerical :
                num_list.append(crt_elem)
            else :
                # not numerical, exit
                return regress_flag
        def_min_len = 10
        def_min_chk = 4
        def_check_fract = 1.0/def_min_len
        len_num_list = len(num_list)
        if len_num_list >= def_min_len :
            # check whether list appears to be made of continuous values
            ret_tuple = Util.SummaryFreqCount(num_list)
            crt_types_no, crt_id_list, crt_count_list = ret_tuple
            if(crt_types_no >= def_min_len) :
                fract_idx = int(crt_types_no * def_check_fract)
                chk_idx = min(fract_idx, def_min_chk)
                if chk_idx > 0 :
                    if crt_count_list[chk_idx] == 1 :
                        # majority of values are unique
                        regress_flag = True
        return regress_flag

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def PreprocessMixToNum(in_list_tbl, in_onehot_max_col, in_num_scale_flag) :

        x_attr = Util.ListDataConvert(in_list_tbl)
        row_no = len(x_attr)
        col_no = len(x_attr[0])
        transp_tbl = Util.MatrixTranspose(x_attr)
        new_tbl = []
        insert_list = []
        attr_dict_lst = []
        num_param_lst = []
        for crt_idx in range(col_no) :
            crt_col = transp_tbl[crt_idx]
            ret_data = Util.AnalyzeVectorType(crt_col, None, in_num_scale_flag)
            is_numerical, has_none_flag, out_list, num_lst_svg, num_lst_min, num_lst_max = ret_data
            if not is_numerical :
                # categorical
                one_hot_lst, one_hot_dict = Util.SelectiveOneHotProc(out_list, in_onehot_max_col)
                insert_no = len(one_hot_lst)
                new_tbl = new_tbl + one_hot_lst
                insert_list.append((crt_idx, insert_no))
                attr_dict_lst.append(one_hot_dict)
                num_param_lst.append(None)
            else :
                # numerical
                new_tbl.append(out_list)
                insert_list.append((crt_idx, 1))
                num_param_dict = {'avg': num_lst_svg, 'min': num_lst_min, 'max': num_lst_max}
                attr_dict_lst.append(None)
                num_param_lst.append(num_param_dict)

            MlpWrk.Misc.TracePrint(">> trace - PreprocessMixToNum, new_tbl:", new_tbl)

        ret_tbl = Util.MatrixTranspose(new_tbl)
        return ret_tbl, insert_list, attr_dict_lst, num_param_lst

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def ConvertMixToNum(in_list_tbl, attr_dict_lst, num_param_lst, in_num_scale_flag) :

        x_attr = Util.ListDataConvert(in_list_tbl)
        MlpWrk.Misc.TracePrint(">> trace - ConvertMixToNum, attr_dict_lst:", attr_dict_lst)
        MlpWrk.Misc.TracePrint(">> trace - ConvertMixToNum, num_param_lst:", num_param_lst)

        row_no = len(x_attr)
        col_no = len(x_attr[0])

        transp_tbl = Util.MatrixTranspose(x_attr)
        new_tbl = []
        insert_list = []
        for crt_idx in range(col_no) :
            MlpWrk.Misc.TracePrint(">> trace - ConvertMixToNum, crt_idx:", crt_idx)
            crt_col = transp_tbl[crt_idx]
            if attr_dict_lst[crt_idx] == None :
                # Numerical columns
                crt_num_elem = num_param_lst[crt_idx]
                num_lst_svg = crt_num_elem['avg']
                num_lst_min = crt_num_elem['min']
                num_lst_max = crt_num_elem['max']
                ret_data = Util.AnalyzeVectorType(crt_col, crt_num_elem, in_num_scale_flag)
                is_numerical, has_none_flag, out_list, num_lst_svg, num_lst_min, num_lst_max = ret_data
                new_tbl.append(out_list)
            else :
                crt_convert_col = []
                crt_dict = attr_dict_lst[crt_idx]
                new_col_no = len(crt_dict)
                extra_tbl = [([0] * row_no)] * new_col_no
                for elem_idx in range(row_no) :
                    crt_elem = crt_col[elem_idx]
                    if crt_elem in crt_dict :
                        rel_col_idx = crt_dict[crt_elem]
                        extra_tbl[rel_col_idx][elem_idx] = 1
                new_tbl = new_tbl + extra_tbl

        ret_tbl = Util.MatrixTranspose(new_tbl)
        return ret_tbl

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# >-----------------------------------------------------------------------------
# >-----------------------------------------------------------------------------

#""" #

# >-----------------------------------------------------------------------------
# >-----------------------------------------------------------------------------

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def UnitTestMlpup() :

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    utest_test_no = 0
    utest_fail_counter = 0

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    print ("- - - - ")
    print ("- - test predict")
    print ("- - - - ")

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    test_list = []

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    ref_param_dict = {  'no_epochs': 100,
                        'learn_rate': 0.1,
                        'hid_dim_lst': [3, 2],
                        'activation': 'sigmoid',
                        # 'activation': 'tanh',
                        'loss': 'mse',
                        # 'loss': 'mae',
                        'pattern_init': False,
                        'predict_mode': 'auto',
                        # 'predict_mode': 'regress',
                        # 'predict_mode': 'classif',
                        'trace_cfg': {
                            'trace_enable': True,
                            'epoch_period': 1,
                        },
                        'onehot_limit': 0,
                        'scale_input': True,
                    }

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    tr_X =  [
                ['x', 'x'],
                ['x', 'y'],
                ['y', 'x'],
                ['y', 'y'],
            ]

    tr_y =  [
                0,
                1,
                1,
                0,
            ]

    param_dict = {
                    'predict_mode': 'auto',
                    'hid_dim_lst': [3, 2],
                    'no_epochs': 20000,
                    'trace_cfg': {
                        'trace_enable': True,
                        'epoch_period': 1000,
                    },
                    'onehot_limit': 0,
                }

    test_list.append((tr_X, tr_y, param_dict))

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    tr_X =  [
                ['x', 'x'],
                ['x', 'y'],
                ['y', 'x'],
                ['y', 'y'],
            ]

    tr_y =  [
                'False',
                'True',
                'True',
                'False',
            ]

    param_dict = {
                    'predict_mode': 'classif',
                    'hid_dim_lst': [3, 2],
                    'no_epochs': 20000,
                    'trace_cfg': {
                        'trace_enable': True,
                        'epoch_period': 1000,
                    },
                    'onehot_limit': 0,
                }

    test_list.append((tr_X, tr_y, param_dict))

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    tr_X =  [
                ['x', 'x'],
                ['x', 'y'],
                ['y', 'x'],
                ['y', 'y'],
            ]

    tr_y =  [
                -100,
                100,
                100,
                -100,
            ]

    param_dict = {
                    'predict_mode': 'regress',
                    'hid_dim_lst': [3, 2],
                    'no_epochs': 30000,
                    'trace_cfg': {
                        'trace_enable': True,
                        'epoch_period': 5000,
                    },
                    'onehot_limit': 0,
                }

    test_list.append((tr_X, tr_y, param_dict))

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    test_elem_no = len(test_list)
    for crt_idx in range(test_elem_no) :

        crt_pair = test_list[crt_idx]
        tr_X = crt_pair[0]
        tr_y = crt_pair[1]
        tr_param = crt_pair[2]

        tt_t =  tr_X[:]
        tt_e =  tr_y[:]

        print ("- - - - tr_X:", tr_X)
        print ("- - - - tr_y:", tr_y)
        print ("- - - - tt_t:", tt_t)
        print ("- - - - tt_e:", tt_e)

        aux_param_dict = dict(ref_param_dict)
        Util.UpdateDict(aux_param_dict, tr_param)
        print ("- - - - aux_param_dict:", aux_param_dict)

        print ("- - - - ")
        print ("- - - - create predictive object")
        tt_o = MlpUnivPredict(aux_param_dict)
        print ("- - - - fit")
        tt_o.fit(tr_X, tr_y)
        print ("- - - - predict")
        tt_predict = tt_o.predict(tt_t)
        print ("- - - - ")

        test_expect = tt_e
        test_result = tt_predict

        print ("- - - - test_expect:", test_expect)
        print ("- - - - test_result:", test_result)
        print ("- - - - ")

        set_eval = ( test_result == test_expect )
        print ("- - - - utest_test_no:", utest_test_no)
        utest_test_no += 1
        if set_eval :
            print ("- - - -   test ok")
        else :
            print ("- - - -  test failed")
            utest_fail_counter += 1
            print ("- - - -  invalid test_result")

            print ("- - - Unit test failure !")
        print ("- - - - ")

    print ("- - total tests:", utest_test_no)
    print ("- - fail tests:", utest_fail_counter)
    print ("- - - - ")

#""" #

if __name__ == "__main__":

    # >-----------------------------------------------------------------------------
    Util.SepLine()
    Util.CrtTimeStamp()
    Util.SepLine()
    # >-----------------------------------------------------------------------------

    UnitTestMlpup()

    # >-----------------------------------------------------------------------------
    Util.SepLine()
    Util.CrtTimeStamp()
    Util.SepLine()
    # >-----------------------------------------------------------------------------

# >-----------------------------------------------------------------------------
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# >-----------------------------------------------------------------------------
