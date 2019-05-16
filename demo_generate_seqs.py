"""A very simple demo showing how to use Bouncing MNIST."""

from data.moving_mnist.data_handler import ChooseDataHandler
from data.moving_mnist.data_handler import ReadDataProto

demo_type = 'train'

if demo_type == 'train':
  data_proto = '/usr/local/projects/prediction/sequence_prediction/moving_mnist/datasets/bouncing_mnist.pbtxt'
elif demo_type == 'validation':
  data_proto = '/usr/local/projects/prediction/sequence_prediction/moving_mnist/datasets/bouncing_mnist_valid.pbtxt'
else:
  raise ValueError('Unknown demo_type: {}.'.format(demo_type))

train_data = ChooseDataHandler(ReadDataProto(data_proto))

# To dequeue from here:
raw_seqs, _ = train_data.GetBatch()

# Here, we use x as the true, reconstruction, and future sequences to
# demonstrate what's expected.
seqs_reshaped = raw_seqs.reshape([train_data.GetBatchSize(),
                                  train_data.GetSeqLength(), 64, 64])
n_input = 10
n_output = train_data.GetSeqLength() - n_input
reconstructed_seqs = seqs_reshaped[:, :n_input, :, :]
future_seqs = seqs_reshaped[:, n_input:, :, :]
case_id = 0  # Display the 0th sequence in the batch

train_data.DisplayData(raw_seqs, rec=reconstructed_seqs,
                       fut=future_seqs, case_id=case_id)
