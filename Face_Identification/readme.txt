一. 运行本项目需要有如下module
dbutils
pymysql
PyQt5
numpy
datetime
cv2
如果本地没有如上库,请先安装,命令如下:
pip install dbutils
二. 本地创建mysql数据库

1.数据库按照如下创建
create database FaceIdentity;
# 员工信息表
create table employeeinfo (
    id int primary key auto_increment,
    ename varchar(50) not null,
    dname varchar(50) not null,
    hidrdate datetime not null
) auto_increment = 10001;
# 打卡表
create table attendancerecord (
    recordid int primary key auto_increment,
    employeeid int not null,
    checkintime datetime not null
);
# 离职表
create table ex_employeeinfo(
    employeeid int not null,
    ename varchar(50) not null,
    dname varchar(50) not null,
    departure_date datetime not null
)
2. 数据库访问用户和密码,改为本地的访问用户和密码,一般为root用户
    设置文件位于ui_resource/DB.py

三. 修改FaceAtten.py和Manager.py中的self.classifier_path
    将路径修改为本地python环境变量该文件位置,如下为本机python环境中的位置
    D:/python3.9.10/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml