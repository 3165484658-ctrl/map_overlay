编译方式，
放入工作空间src目录下
修改以下三个节点的地图文件地址，改为自己项目里的地址。
config/map_overlay_params.yaml
launch/overlay_mapping.launch.py
launch/localization.launch.py

然后安装编译
colcon build --packages-select map_overlay && source install/setup.bash
按提示安装依赖后运行启动指令

启动（定位阶段，不建图）
ros2 launch map_overlay overlay_mapping.launch.py map_id:=70.yaml

初始amcl定位后启动建图服务
ros2 service call /map_overlay/start_mapping std_srvs/srv/Trigger
暂停建图服务
ros2 service call /map_overlay/stop_mapping std_srvs/srv/Trigger
保存地图服务
ros2 service call /map_overlay/save_map std_srvs/srv/Trigger
