# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: event_frame.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x11\x65vent_frame.proto"(\n\x05\x45vent\x12\t\n\x01x\x18\x01 \x01(\x05\x12\t\n\x01y\x18\x02 \x01(\x05\x12\t\n\x01p\x18\x03 \x01(\x05"W\n\nEventFrame\x12\x12\n\ntime_stamp\x18\x01 \x01(\x05\x12\x0e\n\x06height\x18\x02 \x01(\x05\x12\r\n\x05width\x18\x03 \x01(\x05\x12\x16\n\x06\x65vents\x18\x04 \x03(\x0b\x32\x06.Event'
)

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, "event_frame_pb2", globals())
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    _EVENT._serialized_start = 21
    _EVENT._serialized_end = 61
    _EVENTFRAME._serialized_start = 63
    _EVENTFRAME._serialized_end = 150
# @@protoc_insertion_point(module_scope)
