<template>
  <a-space orientation="vertical" size="large" style="display: flex;">
    <!-- 欢迎部分 -->
    <div class="welcome-block">
      <div class="welcome-content">
        <span style="font-size: 16px">你好，很高兴认识你👋</span>
        <span style="font-size: 30px; font-weight: bold; margin: 10px 0">我叫 罗浩森</span>
        <span style="font-size: 16px">是一位 生信分析工程师、前/后端开发初学者</span>
      </div>
    </div>

    <!-- 人格类型 -->
    <div class="personality-type-block">
      <!-- 人格类型说明 -->
      <div class="personality-type-block-content">
        <div style="display: flex; flex-direction: column;">
          <span style="color: gray; font-size: 25px;">人格类型</span>
          <span style="font-size: 50px; font-weight: bold; color: #f0b100">ISTP-A</span>
          <span style="font-size: 30px; color: #f0b100">鉴赏家</span>
        </div>
      </div>
      <!-- 人格图片 -->
      <a class="personality-link" href="https://www.16personalities.com/ch/istp-%E4%BA%BA%E6%A0%BC" target="_blank">
        <img class="istp-img-style" src="/about/ISTP-A.svg" />
      </a>
    </div>

    <!-- 教育背景 -->
    <div>
      <div style="font-size: 30px; font-weight: bold; margin: 10px 0">教育背景</div>
    </div>

    <!-- 技术栈部分 -->
    <div>
      <div style="font-size: 30px; font-weight: bold; margin: 10px 0">技术栈</div>
    </div>

    <!-- 个人职业生涯纪录 -->
    <div>
      <div style="font-size: 30px; font-weight: bold; margin: 10px 0">个人职业生涯纪录</div>
    </div>

    <!-- 个人游玩时间 -->
    <div>
      <div style="font-size: 30px; font-weight: bold; margin: 10px 0">个人游玩时间</div>
    </div>

    <!-- 欢迎通过以下方式联系我 -->
    <div>
      <div style="font-size: 30px; font-weight: bold; margin: 10px 0">欢迎通过以下方式联系我</div>
    </div>
  </a-space>
</template>

<script setup lang="ts">

defineOptions({
  name: 'AboutPageVue'
})

</script>

<style scoped>
/* 欢迎块 */
.welcome-block {
  display: flex;
  align-items: center;
  background: linear-gradient(40deg, rgb(58, 58, 235), cyan);
  padding: 20px;
  border-radius: 10px;
  height: 150px;
}

/* 欢迎内容统一样式 */
.welcome-content {
  display: flex;
  flex-direction: column;
  color: #fff;
  line-height: 35px;
}

/* 人格类型块 */
.personality-type-block {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 225px;
  background: #fff;
  border-radius: 10px;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}

/* 人格类型块内容统一样式 */
.personality-type-block-content {
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: space-between;
}

/* 人格图片样式 */
.istp-img-style {
  height: 200px;
  animation: swing 2s ease-in-out infinite;
}

/* 去掉a标签的after伪类样式 */
.personality-link::after {
  display: none !important;
}

/* 人格类型图片左右摆动动画 */
@keyframes swing {

  0%,
  100% {
    transform: rotate(5deg);
  }

  50% {
    transform: rotate(-5deg);
  }
}
</style>