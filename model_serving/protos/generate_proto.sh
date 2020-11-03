python3 -m grpc_tools.protoc --proto_path=. ./predict.proto --python_out=. --grpc_python_out=.

sed -i '/import predict_pb2 as predict__pb2/c\from . import predict_pb2 as predict__pb2' ./predict_pb2_grpc.py
