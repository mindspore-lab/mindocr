# MindOCR 贡献指南

欢迎投稿，非常感谢！每一点帮助都有作用，功劳也总是会被认可的。

## 贡献者许可协议

在您首次向 MindOCR 社区提交代码之前，需要签署 CLA。

对于个人贡献者，请参阅[ICLA在线文档](https://www.mindspore.cn/icla)以获取详细信息。

## 贡献类型

### 报告 Bugs

在 https://github.com/mindspore-lab/mindocr/issues 报告bug。

如果您要报告bug，请包括：

* 您的操作系统名称和版本。
* 有关您的本地设置的任何详细信息，可能有助于故障排除。
* 重现错误的详细步骤。

### 修复 Bugs

查看GitHub问题中的错误。任何标记为“bug”和“help wanted”的内容，都向任何想要修复它的人开放。

### 实现特性

查看 GitHub 问题以获取功能。任何带有“enhancement”和“help wanted”的标签，都对任何想要实现它的人开放。

### 编写文档

MindOCR总是可以使用更多的文档，无论是作为官方MindOCR文档的一部分，在文档字符串中，还是在博客文章、文章等网络资源中。

### 提交反馈

发送反馈的最佳方式是向 https://github.com/mindspore-lab/mindocr/issues 提交问题。

如果您正在提议一项功能：

* 详细解释它是如何工作的。
* 尽可能缩小范围，以便于实施。
* 请记住，这是一个志愿者驱动的项目，欢迎做出贡献:)

## 入门

准备好做出贡献了吗？以下是为本地开发设置“mindocr”的方法。

1. 在 [GitHub](https://github.com/mindspore-lab/mindocr)上fork 'mindocr' 仓库。
2. 在本地克隆你的 fork：

   ```shell
   git clone git@github.com:your_name_here/mindocr.git
   ```

   之后，您应该将官方仓库添加为上游仓库：

   ```shell
   git remote add upstream git@github.com:mindspore-lab/mindocr
   ```

3. 将本地副本安装到 conda 环境中。假设你已经安装了 conda，以下是你如何设置你的分支进行本地开发的方法：

   ```shell
   conda create -n mindocr python=3.8
   conda activate mindocr
   cd mindocr
   pip install -e .
   ```

4. 创建本地开发分支：

   ```shell
   git checkout -b name-of-your-bugfix-or-feature
   ```

   现在，您可以在本地进行更改。

5. 完成更改后，请检查您的更改是否通过了测试：

   ```shell
   pre-commit run --show-diff-on-failure --color=always --all-files
   pytest
   ```

   如果所有静态测试都通过了，您将得到如下输出：

   ![提交成功前](https://user-images.githubusercontent.com/74176172/221346245-ea868015-bb09-4e53-aa56-73b015e1e336.png)

   否则，您需要根据输出修复警告：

   ![提交前失败](https://user-images.githubusercontent.com/74176172/221346251-7d8f531f-9094-474b-97f0-fd5a55e6d3de.png)

   要获取 pre-commit 和 pytest，只需将它们 pip 安装到您的 conda 环境中。

6. 提交您的更改并将您的分支推送到 GitHub：

   ```shell
   git add .
   git commit -m "你更改内容的详细描述"
   git push origin name-of-your-bugfix-or-feature
   ```

7. 通过 GitHub 网站提交拉取请求。

## 拉取请求指南

在提交拉取请求之前，请检查它是否满足以下准则：

1. 拉取请求应包含tests。
2. 如果拉取请求添加了功能，则应更新文档。将你的新功能放入一个带有文档字符串的函数中，并将该功能添加到README.md中的列表中。
3. 拉取请求应该适用于 Python 3.7、3.8 和 3.9，以及 PyPy。检查 https://github.com/mindspore-lab/mindocr/actions
   并确保所有受支持的 Python 版本的测试都通过。

## 小贴士

您可以安装 git hook 脚本，而不是手动使用 'pre-commit run -a' 进行 linting。

运行以下命令设置 Git Hook 脚本

```shell
pre-commit install
```

现在 'pre-commit' 将在 'git commit' 上自动运行！

## 发布

提醒维护人员如何部署。
请确保已提交所有更改。
然后运行：

```shell
bump2version patch # possible: major / minor / patch
git push
git push --tags
```

如果测试通过，GitHub Action 将部署到 PyPI。
