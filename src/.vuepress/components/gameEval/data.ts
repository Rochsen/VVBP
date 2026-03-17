import { h } from "vue";

// 游戏评价接口
export interface GameEvaluation {
  gameName: string;
  gameCategory: string[];
  gameTime: string;
  myScore: number;
  key: string;
}

// 表格列名
export const gameEvalColumns = [
  {
    title: "游戏名称",
    dataIndex: "gameName",
    key: "gameName",
    width: 250,
  },
  {
    title: "游戏分类",
    dataIndex: "gameCategory",
    key: "gameCategory",
    width: 200,
    render: (_, record) => {
      return record.gameCategory.map((category, index) =>
        h(
          "span",
          {
            key: index,
            class: "category-tag",
          },
          category,
        ),
      );
    },
  },
  {
    title: "游戏时间",
    dataIndex: "gameTime",
    key: "gameTime",
    width: 150,
    sorter: {
      compare: (a, b) => a.gameTime - b.gameTime,
      multiple: 1,
    },
  },
  {
    title: "我的评分",
    dataIndex: "myScore",
    key: "myScore",
    width: 150,
    sorter: {
      compare: (a, b) => a.myScore - b.myScore,
      multiple: 2,
    },
    render: (_, record) => {
      return h(
        "span",
        {
          class: "score-tag",
          style: {
            color:
              record.myScore >= 90
                ? "#52c41a"
                : record.myScore >= 80
                  ? "#1890ff"
                  : record.myScore >= 70
                    ? "#faad14"
                    : "#f5222d",
          },
        },
        record.myScore,
      );
    },
  },
];

// 个人游戏评价列表
export const gameEvalList: GameEvaluation[] = [
  {
    gameName: "艾尔登法环·黑夜君临",
    gameCategory: ["多人合作", "肉鸽"],
    gameTime: "885.5",
    myScore: 90,
    key: "2622380",
  },
  {
    gameName: "艾尔登法环",
    gameCategory: ["类魂"],
    gameTime: "196",
    myScore: 90,
    key: "1245620",
  },
  {
    gameName: "暖雪",
    gameCategory: ["肉鸽"],
    gameTime: "136.3",
    myScore: 80,
    key: "1296830",
  },
  {
    gameName: "黑暗之魂3",
    gameCategory: ["魂类"],
    gameTime: "124.8",
    myScore: 95,
    key: "374320",
  },
  {
    gameName: "黑神话·悟空",
    gameCategory: ["动作"],
    gameTime: "114.8",
    myScore: 95,
    key: "2358720",
  },
  {
    gameName: "雀魂麻将",
    gameCategory: ["益智", "恐怖"],
    gameTime: "110.8",
    myScore: 75,
    key: "1329410",
  },
  {
    gameName: "剑星",
    gameCategory: ["动作"],
    gameTime: "108.5",
    myScore: 95,
    key: "3489700",
  },
  {
    gameName: "Sudoku Zenkai",
    gameCategory: ["益智"],
    gameTime: "67.6",
    myScore: 80,
    key: "809850",
  },
  {
    gameName: "背包乱斗·福西法的宝藏",
    gameCategory: ["益智", "牌组构筑"],
    gameTime: "65.6",
    myScore: 85,
    key: "2427700",
  },
  {
    gameName: "逃离鸭科夫",
    gameCategory: ["平台射击"],
    gameTime: "58.5",
    myScore: 90,
    key: "3167020",
  },
  {
    gameName: "无限机兵",
    gameCategory: ["类魂"],
    gameTime: "33.8",
    myScore: 93,
    key: "2407270",
  },
  {
    gameName: "中国式相亲",
    gameCategory: ["模拟经营"],
    gameTime: "24.7",
    myScore: 85,
    key: "2103130",
  },
  {
    gameName: "叛逆神魂",
    gameCategory: ["牌组构筑", "Galgame"],
    gameTime: "23.9",
    myScore: 85,
    key: "1213300",
  },
  {
    gameName: "火山的女儿",
    gameCategory: ["模拟经营"],
    gameTime: "18.9",
    myScore: 90,
    key: "1669980",
  },
  {
    gameName: "苍翼·混沌效应",
    gameCategory: ["肉鸽"],
    gameTime: "18.2",
    myScore: 90,
    key: "2273430",
  },
  {
    gameName: "明末·渊虚之羽",
    gameCategory: ["类魂"],
    gameTime: "14.9",
    myScore: 90,
    key: "2277560",
  },
  {
    gameName: "盛世天下·媚娘篇",
    gameCategory: ["历史", "互动影视"],
    gameTime: "13.8",
    myScore: 92,
    key: "3478050",
  },
  {
    gameName: "底特律·化身为人",
    gameCategory: ["互动影视"],
    gameTime: "13.2",
    myScore: 95,
    key: "1222670",
  },
  {
    gameName: "江山北望",
    gameCategory: ["互动影视"],
    gameTime: "13.1",
    myScore: 93,
    key: "3831120",
  },
  {
    gameName: "药剂工艺·炼金模拟器",
    gameCategory: ["模拟经营"],
    gameTime: "12.7",
    myScore: 85,
    key: "1210320",
  },
  {
    gameName: "恋爱从离别开始",
    gameCategory: ["互动影视"],
    gameTime: "11.4",
    myScore: 91,
    key: "3652650",
  },
  {
    gameName: "完蛋我被美女包围了2",
    gameCategory: ["互动影视"],
    gameTime: "11.1",
    myScore: 90,
    key: "3282390",
  },
  {
    gameName: "Balatro",
    gameCategory: ["牌组构筑"],
    gameTime: "10.9",
    myScore: 90,
    key: "2379780",
  },
  {
    gameName: "飞跃十三号房",
    gameCategory: ["互动影视"],
    gameTime: "10.6",
    myScore: 94,
    key: "2095300",
  },
  {
    gameName: "无尽星火",
    gameCategory: ["肉鸽"],
    gameTime: "10.2",
    myScore: 90,
    key: "3067890",
  },
  {
    gameName: "你好，我们还有场恋爱没谈",
    gameCategory: ["互动影视"],
    gameTime: "10",
    myScore: 92,
    key: "3167180",
  },
  {
    gameName: "我在地府打麻将",
    gameCategory: ["牌组构筑"],
    gameTime: "9.7",
    myScore: 93,
    key: "3444020",
  },
  {
    gameName: "美女请别影响我成仙",
    gameCategory: ["互动影视"],
    gameTime: "9.5",
    myScore: 91,
    key: "3545990",
  },
  {
    gameName: "疑案追声",
    gameCategory: ["益智", "解谜"],
    gameTime: "8.9",
    myScore: 82,
    key: "942970",
  },
  {
    gameName: "美女请别影响我学习",
    gameCategory: ["互动影视"],
    gameTime: "8.6",
    myScore: 95,
    key: "2786680",
  },
  {
    gameName: "情感反炸模拟器",
    gameCategory: ["互动影视"],
    gameTime: "8.2",
    myScore: 91,
    key: "3350200",
  },
  {
    gameName: "甜心AI追捕计划",
    gameCategory: ["互动影视"],
    gameTime: "8.2",
    myScore: 85,
    key: "2812850",
  },
  {
    gameName: "There is no game",
    gameCategory: ["解谜"],
    gameTime: "8",
    myScore: 95,
    key: "1240210",
  },
  {
    gameName: "我的同事不对劲",
    gameCategory: ["互动影视"],
    gameTime: "7.7",
    myScore: 83,
    key: "3370100",
  },
  {
    gameName: "荒岛求生: 逃出美女岛",
    gameCategory: ["互动影视"],
    gameTime: "7.4",
    myScore: 70,
    key: "3531160",
  },
  {
    gameName: "咸鱼殿下",
    gameCategory: ["互动影视"],
    gameTime: "7.3",
    myScore: 92,
    key: "3433810",
  },
  {
    gameName: "阿西,美女室友竟然...",
    gameCategory: ["互动影视"],
    gameTime: "6.8",
    myScore: 91,
    key: "3021100",
  },
  {
    gameName: "心动恋旅: 樱花篇",
    gameCategory: ["互动影视"],
    gameTime: "6.3",
    myScore: 91,
    key: "3676850",
  },
  {
    gameName: "恶魔轮盘",
    gameCategory: ["益智", "牌组构筑"],
    gameTime: "6.2",
    myScore: 92,
    key: "2835570",
  },
  {
    gameName: "泡芙爱情故事",
    gameCategory: ["Galgame"],
    gameTime: "5.7",
    myScore: 90,
    key: "3389860",
  },
  {
    gameName: "Wallpaper Engine",
    gameCategory: ["工具"],
    gameTime: "939.4",
    myScore: 100,
    key: "431960",
  },
];
