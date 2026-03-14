<template>
  <a-space orientation="vertical" size="large" style="display: flex;">
    <!-- 欢迎部分 -->
    <div class="welcome-block">
      <div class="welcome-content">
        <span style="font-size: 18px">你好，很高兴认识你👋</span>
        <span style="font-size: 30px; font-weight: bold; margin: 10px 0">我叫 罗浩森</span>
        <span style="font-size: 16px">是一位 生物信息分析/前端开发工程师</span>
      </div>
    </div>

    <!-- 人格类型 -->
    <div class="personality-type-block">
      <!-- 人格类型说明 -->
      <div class="personality-type-block-content">
        <div style="display: flex; flex-direction: column; align-items: center;">
          <span class="personality-type-name" style="color: gray; font-size: 25px;">人格类型</span>
          <span class="personality-type-code" style="font-size: 45px; font-weight: bold; color: #f0b100">ISTP-A</span>
          <span class="personality-type-name" style="font-size: 30px; color: #f0b100">鉴赏家</span>
        </div>
      </div>
      <!-- 人格图片 -->
      <a class="personality-link" href="https://www.16personalities.com/ch/istp-%E4%BA%BA%E6%A0%BC" target="_blank">
        <img class="istp-img-style" src="/about/ISTP-A.svg" />
      </a>
    </div>

    <!-- 技术栈部分 -->
    <div>
      <div style="font-size: 30px; font-weight: bold; margin: 10px 0">技术栈</div>
    </div>

    <!-- 教育背景 -->
    <div>
      <div style="font-size: 30px; font-weight: bold; margin: 10px 0">教育背景</div>
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
import "./index.scss"

defineOptions({
  name: 'AboutPageVue'
})

</script>