import { defineClientConfig } from "vuepress/client";
// 全量引入element-plus
// import ElementPlus from "element-plus";
// import "element-plus/dist/index.css";

// 全量引入Antdv-next
import AntdvNext from "antdv-next";
import "antdv-next/dist/antd.css";

// 自定义布局（后续再测试）
// import Layout from "./layouts/Layout.vue";

export default defineClientConfig({
  // layouts: {
  //   Layout
  // },
  enhance({ app }) {
    // 注册AntdV-next
    app.use(AntdvNext);
  },
});
