import { arraySidebar } from "vuepress-theme-hope";

export const zhihuAi = arraySidebar([
  "",
  {
    text: "现在的AI是什么",
    children: [
      "1techOfBigModel",
      "2promptEngToRag",
      "3agentfromControl2selfthink",
      "4multimodalAI",
      "5aiNameProduce",
    ],
  },
  {
    text: "LangChain基础",
    children: ["6langchainApp", "7langchainDesign", "8langchainOptimize"],
  },
  {
    text: "深度学习基础",
    children: ["9networkbaseTensorflow", "10cnnPytorch"],
  },
  {
    text: "LLM模型微调",
    children: [
      "12llmFineTune",
      "13llmFineTuneEval",
      "14llmFineTunePr",
      "15cvandMultiModal",
      "16aiChecK",
    ],
  },
  {
    text: "LLM模型部署",
    children: [
      "18aiDeployCompany",
      "19aiServiceCore",
      "20SQLangOpt",
      "21aiCozeExample",
      "22aiDifyDeploy",
    ],
  },
]);
