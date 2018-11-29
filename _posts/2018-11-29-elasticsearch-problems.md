---
layout: post
title: "elasticsearch 中的问题"
date: 2018-11-29 17:41:09 +0000
categories: es
---

#### 1、can not run elasticsearch as root

	elasticsearch 不能使用root用户安装，需要新建一个用户：

```shell
adduser es //添加用户
passwd es //设置密码
chown es elasticsearch/ -R //修改elasticsearch 所在目录及文件的所有者
su es //以es用户进入
```

#### 2、bootstrap checks failed

	用root用户修改：

		/etc/security/limits.conf

	插入：

```
*          soft      nofile       65536
*          hard      nofile       65537
*          soft      nproc       65536
*          hard      nproc       65537
```

 	修改：

		/etc/sysctl.conf

```shell
vm.max_map_count = 655360
```

	最后执行：

```shell
sysctl -p
```

#### 3、tips

​	获取一个进程的pid值：

```shell
ps -aef|grep service|cut -c 4-10
```

