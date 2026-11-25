# 基线问卷 · Java 后端示例

> 目的：快速建立你的读者模型基线。不用打字，只填数字。

## 怎么填

在每个词后面填 `0` / `1` / `2` / `3`：

| 数字 | 含义 |
|---|---|
| `0` | 不知道 —— 没听过，或听过但完全说不清 |
| `1` | 听过 —— 知道大概是干什么的，没实践过 |
| `2` | 会用 —— 日常写代码/排查会用到，不用查基础资料 |
| `3` | 能讲 —— 能给别人讲清原理、边界和坑 |

**不确定就填 1，宁可低估不要高估。**

---

## 1. Java 语言与并发

```text
Java 语法（集合/泛型/异常/Stream）   __
HashMap（结构/扩容/8转红黑树）          __
ConcurrentHashMap（CAS/size 语义）      __
线程池（七参数/拒绝策略/队列）          __
synchronized（锁升级/对象头）           __
volatile（可见性/禁止重排）             __
CAS（ABA/自旋代价）                     __
AQS（state/CLH 队列）                   __
ThreadLocal（内存泄漏/线程池污染）      __
CompletableFuture（组合/异常）          __
JMM（happens-before）                   __
GC 调优（参数/日志/停顿）               __
```

## 2. JVM

```text
内存区域（堆/栈/方法区/元空间）         __
垃圾回收（分代/回收算法）               __
G1（Region/Mixed GC/停顿预测）          __
Full GC（触发条件/排查）                __
类加载（加载-验证-准备-解析-初始化）    __
双亲委派（ClassLoader 结构）            __
JIT（C1/C2/内联/逃逸分析）              __
safepoint（STW 与安全点）               __
字节码（class 结构/ASM）                __
jmap / jstack（dump/线程栈）            __
arthas（线上排查）                      __
```

## 3. Spring / Spring Boot

```text
IOC（容器/BeanDefinition）              __
AOP（代理/切点/通知）                   __
Bean 生命周期（实例化到销毁）           __
循环依赖（三级缓存）                    __
事务传播（REQUIRED/REQUIRES_NEW/NESTED） __
事务失效（自调用/异常吞掉/非 public）   __
Spring MVC（DispatcherServlet/绑定）    __
拦截器 / Filter（顺序/场景）            __
Spring Boot 自动配置（SPI/条件注解）    __
Spring Cloud（Nacos/OpenFeign/Gateway） __
分布式事务（Seata/TCC/本地消息表）      __
Spring Security（过滤器链/认证）        __
```

## 4. MySQL / 数据库

```text
SQL（多表 join/子查询/聚合）            __
索引（B+ 树/聚簇/二级）                 __
联合索引（列顺序）                  __
回表（二级到聚簇）                  __
覆盖索引（避免回表）                __
最左前缀                            __
explain（type/key/rows/Extra）          __
事务隔离级别（RU/RC/RR/SERIALIZABLE）   __
MVCC（Read View/undo log）              __
间隙锁（next-key/幻读）                 __
慢 SQL 优化（定位与改写）               __
分库分表（ShardingSphere）              __
主从复制（binlog/延迟/读写分离）        __
MyBatis（动态 SQL/resultMap/缓存）      __
```

## 5. Redis / 缓存

```text
Redis 数据类型（String/Hash/List/Set/ZSet） __
缓存穿透（空值/布隆）                   __
缓存击穿（互斥重建/逻辑过期）           __
缓存雪崩（TTL 抖动/多级缓存）           __
分布式锁（SET NX PX/续期/fencing）      __
RDB / AOF（持久化代价）                 __
主从 / 哨兵 / Cluster                   __
缓存一致性（先删后删/延迟双删）         __
热 key / 大 key（识别与治理）           __
```

## 6. 消息队列

```text
RocketMQ（架构/刷盘/重试）              __
Kafka（分区/消费组/offset）             __
消息丢失（三段式可靠性）                __
重复消费（幂等设计）                    __
顺序消息（队列选择/单线程）             __
死信队列（重试上限/兜底）               __
事务消息（半消息/回查）                 __
```

## 7. K8s / 运维

```text
Pod（最小调度单位/生命周期）            __
Deployment / Service（滚动/发现）       __
kubectl（常用排查命令）                 __
ConfigMap / Secret（配置注入）          __
探针（liveness/readiness/startup）      __
HPA（指标扩缩容）                       __
Operator / CRD（自定义控制器）          __
Docker（镜像/分层/Dockerfile）          __
Linux（top/grep/awk/网络排查）          __
Shell（脚本编写）                       __
CI/CD（Jenkins/GitLab CI）              __
```

## 8. 工程工具

```text
Git（分支/merge/rebase/冲突）           __
Maven / Gradle（依赖冲突/多模块）       __
JUnit / Mockito（单测）                 __
日志框架（SLF4J/Logback/MDC）           __
接口文档（Swagger/Knife4j）             __
压力测试（JMeter/wrk/P99）              __
```

---

## 填完之后

把这份文件填好保存，然后告诉我，我会用脚本导入 `var/reader`。

如果想补充别的领域（Python / 前端 / 算法 / 大数据），先在 JSON 的 `groups` 里加一个同 `id` 的组，再在最下面加一节，格式同上。
