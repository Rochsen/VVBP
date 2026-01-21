import { navbar } from "vuepress-theme-hope";

export default navbar([
  // 主页
  "/",
  // 笔记
  "/notes/",
  // 关于
  "/about/",
  // 友链
  "/friendlinks/",
  // 攻略
  {
    text: "攻略",
    icon: "pen-to-square",
    prefix: "/strategy/",
    children: [
      {
        text: "艾尔登法环 黑夜君临",
        icon: "person-running",
        prefix: "nightReign/",
        children: [
          {
            text: "Boss弱点",
            icon: "person-running",
            link: "bossAnti",
          },
          {
            text: "第三晚Boss推断",
            icon: "person-running",
            link: "thirdNightBossInfer",
          },
          {
            text: "大空洞水晶位置",
            icon: "person-running",
            link: "bigHoleCrystal",
          },
          {
            text: "流派整理",
            icon: "person-running",
            link: "genre",
          },
        ],
      },
    ],
  },
  // example from hope
  // "/demo/",
  // {
  //   text: "博文",
  //   icon: "pen-to-square",
  //   prefix: "/posts/",
  //   children: [
  //     {
  //       text: "苹果",
  //       icon: "pen-to-square",
  //       prefix: "apple/",
  //       children: [
  //         { text: "苹果1", icon: "pen-to-square", link: "1" },
  //         { text: "苹果2", icon: "pen-to-square", link: "2" },
  //         "3",
  //         "4",
  //       ],
  //     },
  //     {
  //       text: "香蕉",
  //       icon: "pen-to-square",
  //       prefix: "banana/",
  //       children: [
  //         {
  //           text: "香蕉 1",
  //           icon: "pen-to-square",
  //           link: "1",
  //         },
  //         {
  //           text: "香蕉 2",
  //           icon: "pen-to-square",
  //           link: "2",
  //         },
  //         "3",
  //         "4",
  //       ],
  //     },
  //     { text: "樱桃", icon: "pen-to-square", link: "cherry" },
  //     { text: "火龙果", icon: "pen-to-square", link: "dragonfruit" },
  //     "tomato",
  //     "strawberry",
  //   ],
  // },
  // {
  //   text: "V2 文档",
  //   icon: "book",
  //   link: "https://theme-hope.vuejs.press/zh/",
  // },
]);
