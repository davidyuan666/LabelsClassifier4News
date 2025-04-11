## 标签预测API文档

### 接口路径

```
POST http://10.20.64.3:8009/tags/check
```

### 描述

该 API 接口接收文本内容，返回预测的标签（`p0`、`p1`、`p2`）。

### 请求头

- Content-Type: `application/json`

### 请求体

| 字段            | 类型   | 描述                             |
|---------------| ------ |--------------------------------|
| `contents`    | Array  | 字符串数组，每个字符串代表一段要分析的文本内容。       |
| `request_id`  | String | 请求的接口标识符。                      |
| `content_ids` | Array  | 字符串数组，每个字符串代表一段要分析的文本内容的标号。    |
| `source_type` | String | 内容来源的类型（如 `news`、`article` 等）。 |

#### 请求体示例

```
{
  "contents": ["文本内容1", "文本内容2"],
  "request_id": "123456789",
  "content_ids": ["100001","100002"]
  "source_type": "news"
}
```


### 响应体

| 字段            | 类型      | 描述            |
|---------------|---------|---------------|
| `request_id`  | String  | 回显请求中的唯一标识符。  |
| `source_type` | String  | 回显请求中的内容来源类型。 |
| `content_ids` | Array   | 文本内容标识号码数组。   |
| `result`      | Array   | 对象数组          |

#### 示例

```
{
  "request_id": "001",
  "source_type": "news",
  "content_ids": ["100001", "100002","100003"],
  "result": [
    {
      "p0": [
        {"content": "消息陆风X2这么火爆的车型全国多个城市限安", "prediction": "0"},
        {"content": "消息的车型全国多个城市限时优惠其中安", "prediction": "0"},
        {"content": "消息陆风X2这么火爆的时优惠其中安", "prediction": "0"}
      ]
    },
    // 同理，p1 和 p2 的结果格式相同
  ]
}

```

### 接口路径

```
POST http://10.20.64.3:8009/tags/predict
```

### 描述

该API接口接收文本内容，并根据指定的检查标签（`p0`、`p1`、`p2`）返回预测的标签及其从父节点到叶节点的层级路径。

### 请求头

- Content-Type: `application/json`

### 请求体

| 字段            | 类型   | 描述                         |
|---------------| ------ |----------------------------|
| `contents`    | Array  | 字符串数组，每个字符串代表一段要分析的文本内容。   |
| `content_ids` | Array  | 字符串数组，每个字符串代表一段要分析的文本内容的标号。   |
| `request_id`  | String | 请求的接口标识符。                  |
| `source_type` | String | 内容来源的类型（`news`、`article`）。 |
| `check_tag`   | String | 指定用于预测的模型。可以是`p0`、`p1`、p2。 |

#### 示例

```
{
  "contents": [
    "文本内容1",
    "文本内容2"
  ],
  "content_ids": ["100001","100002"]
  "request_id": "123456789",
  "source_type": "news",
  "check_tag": "p0"
}
```

### 响应体

| 字段          | 类型   | 描述                                                         |
| ------------- | ------ | ------------------------------------------------------------ |
| `request_id`  | String | 回显请求中的唯一标识符。                                     |
| `source_type` | String | 回显请求中的内容来源类型。                                   |
| `check_tag`   | String | 回显请求中指定的预测模型。                                   |
| `content_ids` | Array   | 文本内容标识号码数组。   |
| `label_list`  | Array  | 对象数组，每个对象包含`content`（输入文本）、`labels`（内容的预测标签）和`label_hierarchy_paths`（预测标签的层级路径）。 |

#### 示例

```
{
  "request_id": "123456789",
  "source_type": "news",
  "check_tag": "p0",
  "content_ids": ["100001","100002"]
  "label_list": [
    {
      "content": "文本内容1",
      "labels": ["标签1", "标签2"],
      "label_hierarchy_paths": ["父级,子级,标签1", "父级,子级,标签2"]
    },
    {
      "content": "文本内容2",
      "labels": ["标签3"],
      "label_hierarchy_paths": ["父级,子级,标签3"]
    }
  ]
}
```

### 错误响应

如果请求中缺少任何必需字段，API将响应400状态码，并提供详细的缺少字段错误信息。

#### 示例

```
{
  "error": "缺少必需字段"
}
```

