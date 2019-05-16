# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: config.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='config.proto',
  package='config',
  syntax='proto2',
  serialized_pb=_b('\n\x0c\x63onfig.proto\x12\x06\x63onfig\"\xd7\x04\n\x04\x44\x61ta\x12\x11\n\tdata_file\x18\x01 \x02(\t\x12\x14\n\x0c\x64\x61taset_name\x18\x02 \x02(\t\x12\x13\n\x0blabels_file\x18\x03 \x01(\t\x12\x17\n\x0fnum_frames_file\x18\x04 \x01(\t\x12\x15\n\nnum_frames\x18\x05 \x01(\x05:\x01\x31\x12\x11\n\x06stride\x18\x06 \x01(\x05:\x01\x31\x12\x18\n\trandomize\x18\x07 \x01(\x08:\x05\x66\x61lse\x12\x12\n\nbatch_size\x18\x08 \x01(\x05\x12\x38\n\x0c\x64\x61taset_type\x18\t \x01(\x0e\x32\x18.config.Data.DatasetType:\x08LABELLED\x12\x12\n\nimage_size\x18\n \x01(\x05\x12\x12\n\nnum_digits\x18\x0b \x01(\x05\x12\x13\n\x0bstep_length\x18\x0c \x01(\x02\x12\x16\n\x0evideo_ids_file\x18\r \x01(\t\x12\x17\n\x0cimage_size_x\x18\x0e \x01(\x05:\x01\x30\x12\x17\n\x0cimage_size_y\x18\x0f \x01(\x05:\x01\x30\x12\x17\n\x0cpatch_size_x\x18\x10 \x01(\x05:\x01\x30\x12\x17\n\x0cpatch_size_y\x18\x11 \x01(\x05:\x01\x30\x12\x15\n\nnum_colors\x18\x12 \x01(\x05:\x01\x30\x12\x11\n\tmean_file\x18\x13 \x01(\t\x12\x17\n\x0csample_times\x18\x14 \x01(\x05:\x01\x31\"j\n\x0b\x44\x61tasetType\x12\x0c\n\x08LABELLED\x10\x00\x12\x0e\n\nUNLABELLED\x10\x01\x12\x12\n\x0e\x42OUNCING_MNIST\x10\x02\x12\x18\n\x14\x42OUNCING_MNIST_FIXED\x10\x03\x12\x0f\n\x0bVIDEO_PATCH\x10\x04\"\xc3\x02\n\x05Param\x12)\n\tinit_type\x18\x01 \x01(\x0e\x32\x16.config.Param.InitType\x12\x10\n\x05scale\x18\x02 \x01(\x02:\x01\x30\x12\x12\n\x07\x65psilon\x18\x03 \x01(\x02:\x01\x30\x12\x13\n\x08momentum\x18\x04 \x01(\x02:\x01\x30\x12\x13\n\x08l2_decay\x18\x05 \x01(\x02:\x01\x30\x12\x18\n\rgradient_clip\x18\x06 \x01(\x02:\x01\x30\x12\x1b\n\x10\x65ps_decay_factor\x18\x07 \x01(\x02:\x01\x31\x12\x1a\n\x0f\x65ps_decay_after\x18\x08 \x01(\x05:\x01\x30\x12\x11\n\tfile_name\x18\t \x01(\t\x12\x14\n\x0c\x64\x61taset_name\x18\n \x01(\t\"C\n\x08InitType\x12\x0c\n\x08\x43ONSTANT\x10\x00\x12\x0c\n\x08GAUSSIAN\x10\x01\x12\x0b\n\x07UNIFORM\x10\x02\x12\x0e\n\nPRETRAINED\x10\x03\"\x94\x03\n\x04LSTM\x12\x0c\n\x04name\x18\x01 \x02(\t\x12\x0f\n\x07num_hid\x18\x02 \x02(\x05\x12\x1e\n\x07w_dense\x18\x03 \x01(\x0b\x32\r.config.Param\x12\x1d\n\x06w_diag\x18\x04 \x01(\x0b\x32\r.config.Param\x12\x18\n\x01\x62\x18\x05 \x01(\x0b\x32\r.config.Param\x12\x1e\n\x07w_input\x18\x06 \x01(\x0b\x32\r.config.Param\x12\x1f\n\x08w_output\x18\x07 \x01(\x0b\x32\r.config.Param\x12\x1f\n\x08\x62_output\x18\x08 \x01(\x0b\x32\r.config.Param\x12\x18\n\thas_input\x18\t \x01(\x08:\x05\x66\x61lse\x12\x19\n\nhas_output\x18\n \x01(\x08:\x05\x66\x61lse\x12\x17\n\x08use_relu\x18\x0b \x01(\x08:\x05\x66\x61lse\x12\x15\n\ninput_dims\x18\x0c \x01(\x05:\x01\x30\x12\x16\n\x0boutput_dims\x18\r \x01(\x05:\x01\x30\x12\x19\n\x0einput_dropprob\x18\x0e \x01(\x02:\x01\x30\x12\x1a\n\x0foutput_dropprob\x18\x0f \x01(\x02:\x01\x30\"\x88\x01\n\x06Logreg\x12\x0c\n\x04name\x18\x01 \x02(\t\x12\x12\n\nnum_inputs\x18\x02 \x02(\x05\x12\x13\n\x0bnum_outputs\x18\x03 \x02(\x05\x12\x18\n\x01w\x18\x04 \x01(\x0b\x32\r.config.Param\x12\x18\n\x01\x62\x18\x05 \x01(\x0b\x32\r.config.Param\x12\x13\n\x08\x64ropprob\x18\x06 \x01(\x02:\x01\x30\"\xf5\x04\n\x05Model\x12\x0c\n\x04name\x18\x01 \x02(\t\x12\x1a\n\x04lstm\x18\x02 \x03(\x0b\x32\x0c.config.LSTM\x12\x1e\n\x06logreg\x18\x03 \x02(\x0b\x32\x0e.config.Logreg\x12\x11\n\ttimestamp\x18\x04 \x03(\t\x12\x16\n\x0e\x63heckpoint_dir\x18\x05 \x01(\t\x12\x14\n\tmax_iters\x18\x06 \x01(\x05:\x01\x30\x12\x16\n\x0bprint_after\x18\x07 \x01(\x05:\x01\x31\x12\x19\n\x0evalidate_after\x18\x08 \x01(\x05:\x01\x30\x12\x15\n\nsave_after\x18\t \x01(\x05:\x01\x30\x12\x18\n\rdisplay_after\x18\n \x01(\x05:\x01\x30\x12\x19\n\x0e\x64\x65\x63_seq_length\x18\x0b \x01(\x05:\x01\x30\x12\x1e\n\x08lstm_dec\x18\x0c \x03(\x0b\x32\x0c.config.LSTM\x12\x1a\n\x0bsquash_relu\x18\r \x01(\x08:\x05\x66\x61lse\x12\x1d\n\x12squash_relu_lambda\x18\x0e \x01(\x02:\x01\x32\x12\x1a\n\x0b\x62inary_data\x18\x0f \x01(\x08:\x05\x66\x61lse\x12\x1c\n\x11\x66uture_seq_length\x18\x10 \x01(\x05:\x01\x30\x12!\n\x0blstm_future\x18\x11 \x03(\x0b\x32\x0c.config.LSTM\x12\x1e\n\x0f\x64\x65\x63_conditional\x18\x12 \x01(\x08:\x05\x66\x61lse\x12!\n\x12\x66uture_conditional\x18\x13 \x01(\x08:\x05\x66\x61lse\x12&\n\x17\x64\x65\x63oder_copy_init_state\x18\x14 \x01(\x08:\x05\x66\x61lse\x12%\n\x16\x66uture_copy_init_state\x18\x15 \x01(\x08:\x05\x66\x61lse\x12\x18\n\trelu_data\x18\x16 \x01(\x08:\x05\x66\x61lse')
)



_DATA_DATASETTYPE = _descriptor.EnumDescriptor(
  name='DatasetType',
  full_name='config.Data.DatasetType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='LABELLED', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='UNLABELLED', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BOUNCING_MNIST', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BOUNCING_MNIST_FIXED', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='VIDEO_PATCH', index=4, number=4,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=518,
  serialized_end=624,
)
_sym_db.RegisterEnumDescriptor(_DATA_DATASETTYPE)

_PARAM_INITTYPE = _descriptor.EnumDescriptor(
  name='InitType',
  full_name='config.Param.InitType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='CONSTANT', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GAUSSIAN', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='UNIFORM', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PRETRAINED', index=3, number=3,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=883,
  serialized_end=950,
)
_sym_db.RegisterEnumDescriptor(_PARAM_INITTYPE)


_DATA = _descriptor.Descriptor(
  name='Data',
  full_name='config.Data',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data_file', full_name='config.Data.data_file', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='dataset_name', full_name='config.Data.dataset_name', index=1,
      number=2, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='labels_file', full_name='config.Data.labels_file', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='num_frames_file', full_name='config.Data.num_frames_file', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='num_frames', full_name='config.Data.num_frames', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='stride', full_name='config.Data.stride', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='randomize', full_name='config.Data.randomize', index=6,
      number=7, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='batch_size', full_name='config.Data.batch_size', index=7,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='dataset_type', full_name='config.Data.dataset_type', index=8,
      number=9, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='image_size', full_name='config.Data.image_size', index=9,
      number=10, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='num_digits', full_name='config.Data.num_digits', index=10,
      number=11, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='step_length', full_name='config.Data.step_length', index=11,
      number=12, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='video_ids_file', full_name='config.Data.video_ids_file', index=12,
      number=13, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='image_size_x', full_name='config.Data.image_size_x', index=13,
      number=14, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='image_size_y', full_name='config.Data.image_size_y', index=14,
      number=15, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='patch_size_x', full_name='config.Data.patch_size_x', index=15,
      number=16, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='patch_size_y', full_name='config.Data.patch_size_y', index=16,
      number=17, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='num_colors', full_name='config.Data.num_colors', index=17,
      number=18, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='mean_file', full_name='config.Data.mean_file', index=18,
      number=19, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='sample_times', full_name='config.Data.sample_times', index=19,
      number=20, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _DATA_DATASETTYPE,
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=25,
  serialized_end=624,
)


_PARAM = _descriptor.Descriptor(
  name='Param',
  full_name='config.Param',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='init_type', full_name='config.Param.init_type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='scale', full_name='config.Param.scale', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='epsilon', full_name='config.Param.epsilon', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='momentum', full_name='config.Param.momentum', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='l2_decay', full_name='config.Param.l2_decay', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='gradient_clip', full_name='config.Param.gradient_clip', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='eps_decay_factor', full_name='config.Param.eps_decay_factor', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='eps_decay_after', full_name='config.Param.eps_decay_after', index=7,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='file_name', full_name='config.Param.file_name', index=8,
      number=9, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='dataset_name', full_name='config.Param.dataset_name', index=9,
      number=10, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _PARAM_INITTYPE,
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=627,
  serialized_end=950,
)


_LSTM = _descriptor.Descriptor(
  name='LSTM',
  full_name='config.LSTM',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='config.LSTM.name', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='num_hid', full_name='config.LSTM.num_hid', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='w_dense', full_name='config.LSTM.w_dense', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='w_diag', full_name='config.LSTM.w_diag', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='b', full_name='config.LSTM.b', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='w_input', full_name='config.LSTM.w_input', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='w_output', full_name='config.LSTM.w_output', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='b_output', full_name='config.LSTM.b_output', index=7,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='has_input', full_name='config.LSTM.has_input', index=8,
      number=9, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='has_output', full_name='config.LSTM.has_output', index=9,
      number=10, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='use_relu', full_name='config.LSTM.use_relu', index=10,
      number=11, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='input_dims', full_name='config.LSTM.input_dims', index=11,
      number=12, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='output_dims', full_name='config.LSTM.output_dims', index=12,
      number=13, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='input_dropprob', full_name='config.LSTM.input_dropprob', index=13,
      number=14, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='output_dropprob', full_name='config.LSTM.output_dropprob', index=14,
      number=15, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=953,
  serialized_end=1357,
)


_LOGREG = _descriptor.Descriptor(
  name='Logreg',
  full_name='config.Logreg',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='config.Logreg.name', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='num_inputs', full_name='config.Logreg.num_inputs', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='num_outputs', full_name='config.Logreg.num_outputs', index=2,
      number=3, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='w', full_name='config.Logreg.w', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='b', full_name='config.Logreg.b', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='dropprob', full_name='config.Logreg.dropprob', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1360,
  serialized_end=1496,
)


_MODEL = _descriptor.Descriptor(
  name='Model',
  full_name='config.Model',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='config.Model.name', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='lstm', full_name='config.Model.lstm', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='logreg', full_name='config.Model.logreg', index=2,
      number=3, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='config.Model.timestamp', index=3,
      number=4, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='checkpoint_dir', full_name='config.Model.checkpoint_dir', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='max_iters', full_name='config.Model.max_iters', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='print_after', full_name='config.Model.print_after', index=6,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='validate_after', full_name='config.Model.validate_after', index=7,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='save_after', full_name='config.Model.save_after', index=8,
      number=9, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='display_after', full_name='config.Model.display_after', index=9,
      number=10, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='dec_seq_length', full_name='config.Model.dec_seq_length', index=10,
      number=11, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='lstm_dec', full_name='config.Model.lstm_dec', index=11,
      number=12, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='squash_relu', full_name='config.Model.squash_relu', index=12,
      number=13, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='squash_relu_lambda', full_name='config.Model.squash_relu_lambda', index=13,
      number=14, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(2),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='binary_data', full_name='config.Model.binary_data', index=14,
      number=15, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='future_seq_length', full_name='config.Model.future_seq_length', index=15,
      number=16, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='lstm_future', full_name='config.Model.lstm_future', index=16,
      number=17, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='dec_conditional', full_name='config.Model.dec_conditional', index=17,
      number=18, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='future_conditional', full_name='config.Model.future_conditional', index=18,
      number=19, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='decoder_copy_init_state', full_name='config.Model.decoder_copy_init_state', index=19,
      number=20, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='future_copy_init_state', full_name='config.Model.future_copy_init_state', index=20,
      number=21, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='relu_data', full_name='config.Model.relu_data', index=21,
      number=22, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1499,
  serialized_end=2128,
)

_DATA.fields_by_name['dataset_type'].enum_type = _DATA_DATASETTYPE
_DATA_DATASETTYPE.containing_type = _DATA
_PARAM.fields_by_name['init_type'].enum_type = _PARAM_INITTYPE
_PARAM_INITTYPE.containing_type = _PARAM
_LSTM.fields_by_name['w_dense'].message_type = _PARAM
_LSTM.fields_by_name['w_diag'].message_type = _PARAM
_LSTM.fields_by_name['b'].message_type = _PARAM
_LSTM.fields_by_name['w_input'].message_type = _PARAM
_LSTM.fields_by_name['w_output'].message_type = _PARAM
_LSTM.fields_by_name['b_output'].message_type = _PARAM
_LOGREG.fields_by_name['w'].message_type = _PARAM
_LOGREG.fields_by_name['b'].message_type = _PARAM
_MODEL.fields_by_name['lstm'].message_type = _LSTM
_MODEL.fields_by_name['logreg'].message_type = _LOGREG
_MODEL.fields_by_name['lstm_dec'].message_type = _LSTM
_MODEL.fields_by_name['lstm_future'].message_type = _LSTM
DESCRIPTOR.message_types_by_name['Data'] = _DATA
DESCRIPTOR.message_types_by_name['Param'] = _PARAM
DESCRIPTOR.message_types_by_name['LSTM'] = _LSTM
DESCRIPTOR.message_types_by_name['Logreg'] = _LOGREG
DESCRIPTOR.message_types_by_name['Model'] = _MODEL
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Data = _reflection.GeneratedProtocolMessageType('Data', (_message.Message,), dict(
  DESCRIPTOR = _DATA,
  __module__ = 'config_pb2'
  # @@protoc_insertion_point(class_scope:config.Data)
  ))
_sym_db.RegisterMessage(Data)

Param = _reflection.GeneratedProtocolMessageType('Param', (_message.Message,), dict(
  DESCRIPTOR = _PARAM,
  __module__ = 'config_pb2'
  # @@protoc_insertion_point(class_scope:config.Param)
  ))
_sym_db.RegisterMessage(Param)

LSTM = _reflection.GeneratedProtocolMessageType('LSTM', (_message.Message,), dict(
  DESCRIPTOR = _LSTM,
  __module__ = 'config_pb2'
  # @@protoc_insertion_point(class_scope:config.LSTM)
  ))
_sym_db.RegisterMessage(LSTM)

Logreg = _reflection.GeneratedProtocolMessageType('Logreg', (_message.Message,), dict(
  DESCRIPTOR = _LOGREG,
  __module__ = 'config_pb2'
  # @@protoc_insertion_point(class_scope:config.Logreg)
  ))
_sym_db.RegisterMessage(Logreg)

Model = _reflection.GeneratedProtocolMessageType('Model', (_message.Message,), dict(
  DESCRIPTOR = _MODEL,
  __module__ = 'config_pb2'
  # @@protoc_insertion_point(class_scope:config.Model)
  ))
_sym_db.RegisterMessage(Model)


# @@protoc_insertion_point(module_scope)
