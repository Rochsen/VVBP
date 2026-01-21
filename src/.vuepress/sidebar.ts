import { sidebar } from "vuepress-theme-hope";

export default sidebar({
  "/notes/": "structure",
  "/about/": "structure",
  "/strategy/": [
    {
      text: "艾尔登法环 黑夜君临",
      icon: "book",
      prefix: "nightReign/",
      children: "structure"
    }
  ],
  "/": [
    // "",
    // {
    //   text: "如何使用",
    //   icon: "laptop-code",
    //   prefix: "demo/",
    //   link: "demo/",
    //   children: "structure",
    // },
    // {
    //   text: "文章",
    //   icon: "book",
    //   prefix: "posts/",
    //   children: "structure",
    // },
    // "intro",
    // {
    //   text: "幻灯片",
    //   icon: "person-chalkboard",
    //   link: "https://ecosystem.vuejs.press/zh/plugins/markdown/revealjs/demo.html",
    // },
  ],
});
