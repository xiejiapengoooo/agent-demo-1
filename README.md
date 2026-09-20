## 前端样式

页面使用 Tailwind CSS 和 daisyUI，主题与自定义样式位于 `css/main.css`，构建产物为 `static/main.css`。修改样式或 HTML 中的类名后，在项目根目录执行：

```bash
yarn --cwd css install --frozen-lockfile
yarn --cwd css build
```

开发时可运行 `yarn --cwd css build:watch` 自动重新构建。`static/main.css` 随代码保存，运行后端时无需额外构建。
