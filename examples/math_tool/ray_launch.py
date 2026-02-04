import os
import json

# 解析集群规范
cluster_spec = json.loads(os.environ["AFO_ENV_CLUSTER_SPEC"])
role = cluster_spec["role"]
assert role == "worker", "{} vs worker".format(role)
node_rank = int(cluster_spec["index"])
nnodes = len(cluster_spec[role])
master, ray_port = cluster_spec[role][0].split(":")
master_addr = master
# ray_port = 6379  # Ray的默认端口

# 生成Ray启动命令，每行单独输出，避免多行字符串格式问题
if node_rank == 0:
    print(f"  rank=0;")
    print(f"  ray start --head --port {ray_port}")
    print("sleep 100s ;")
else:
    print(f"  rank={node_rank};")
    print("sleep 100s ;")
    print(f"ray start --address={master_addr}:{ray_port};")
    print("sleep 10000000s ;")
