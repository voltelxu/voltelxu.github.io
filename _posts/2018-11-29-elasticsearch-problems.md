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

	获取一个进程的pid值：

```shell
ps -aef|grep service|cut -c 4-10
```

#### 4、 搜索

	全文搜索：

```js
GET /megacorp/employee/_search
{
    "query" : {
        "match" : {
            "about" : "rock climbing"
        }
    }
}
```

	短语搜索：

```js
GET /megacorp/employee/_search
{
    "query" : {
        "match_phrase" : {
            "about" : "rock climbing"
        }
    }
}
```

#### 5、 故障转移、水平扩容

使用单播代替组播。

**number_of_replicas** ： 每个主分片对应的副分片的个数，对应可以发生故障的 node 的个数（在多个 node 节点时才可以体现出价值），每个主副分片都可用于搜索，若 node 节点多且在不同的节点上都有分片，可以提高搜索性能。

![1544698705(1)](https://voltelxu.github.io/assets/img/1544698705(1).jpg)

![1544698785(1)](https://voltelxu.github.io/assets/img/1544698785(1).jpg)

#### 6、 元数据

\_index，_type， _id 

#### 7、 处理冲突

悲观并发控制

​	这种方法被关系型数据库广泛使用，它假定有变更冲突可能发生，因此阻塞访问资源以防止冲突。 一个典型的例子是读取一行数据之前先将其锁住，确保只有放置锁的线程能够对这行数据进行修改。

乐观并发控制

​	Elasticsearch 中使用的这种方法假定冲突是不可能发生的，并且不会阻塞正在尝试的操作。 然而，如果源数据在读写当中被修改，更新将会失败。应用程序接下来将决定该如何解决冲突。 例如，可以重试更新、使用新的数据、或者将相关情况报告给用户。

```js
PUT /website/blog/1?version=1 
{
  "title": "My first blog entry",
  "text":  "Starting to get the hang of this..."
}
```

