<template>
    <a-input-search v-model:value="inputGameVal" placeholder="搜索游戏名称" style="width: 100%;" :allowClear="true"
        @search="onSearch" @clear="clearSearch">
        <!-- <template #prefix>
            <SearchOutlined />
        </template> -->
    </a-input-search>
    <a-table :columns="gameEvalColumns" :data-source="tableData" :scroll="{ x: 'max-content' }"
        :pagination="paginationConfig" @change="handleTableChange" class="game-table" bordered />
</template>

<script setup lang='ts'>
import { ref } from 'vue'
import { gameEvalList, gameEvalColumns } from './data.ts'
import { SearchOutlined } from '@antdv-next/icons'

defineOptions({
    name: 'gameEvalAppVue'
})

// 响应式数据
const tableData = ref(gameEvalList)

// 搜索框值
const inputGameVal = ref('')


// 搜索框改变时触发
const onSearch = () => {
    console.log("🚀 ~ onSearch ~ inputGameVal.value:", inputGameVal.value)

    // 过滤出包含搜索值的游戏评价
    const filteredList = tableData.value.filter((item) => item.gameName.includes(inputGameVal.value))

    // 更新表格数据
    tableData.value = filteredList
    console.log("🚀 ~ onSearch ~ tableData.value:", tableData.value)

    // 更新分页器配置
    if (inputGameVal.value === '') {
        tableData.value = gameEvalList
    }
    else {
        tableData.value = filteredList
    }

    paginationConfig.value.total = tableData.value.length
}

// 清空搜索框时触发
const clearSearch = () => {
    tableData.value = gameEvalList
}


// 分页器配置
const paginationConfig = ref({
    current: 1,
    pageSize: 10,
    total: tableData.value.length,
    showSizeChanger: true,
    pageSizeOptions: ['10', '20', '50', '100'],
    showTotal: (total: number) => `共 ${total} 条数据`,
    placement: ['bottomCenter']
})

// 分页器改变时触发
const handleTableChange = (pagination: any) => {
    paginationConfig.value.current = pagination.current
    paginationConfig.value.pageSize = pagination.pageSize
}

</script>

<style scoped>
.game-table {
    width: 100%;
    margin-top: 20px;
}

:deep(table) {
    margin: 0
}

/* 确保表格单元格内容正确显示 */
:deep(.ant-table-cell) {
    white-space: normal;
    word-break: break-word;
    vertical-align: middle;
}

/* 分类标签样式 */
:deep(.category-tag) {
    display: inline-block;
    background-color: #f0f0f0;
    border-radius: 4px;
    padding: 2px 8px;
    margin: 2px;
    font-size: 12px;
}

/* 评分标签样式 */
:deep(.score-tag) {
    font-weight: 600;
    font-size: 14px;
}

/* 确保表格头部样式正确 */
:deep(.ant-table-thead > tr > th) {
    font-weight: 600;
    text-align: center;
    background-color: #fafafa;
}

/* 确保表格内容居中 */
:deep(.ant-table-tbody > tr > td) {
    text-align: center;
}

/* 优化表格行 hover 效果 */
:deep(.ant-table-tbody > tr:hover) {
    background-color: #f5f5f5;
}

/* 分页器样式优化 */
:deep(.ant-pagination) {
    margin-top: 20px;
    text-align: center;
}

:deep(.ant-pagination-item) {
    border-radius: 4px;
}

:deep(.ant-pagination-item-active) {
    background-color: #1890ff;
    border-color: #1890ff;
}

:deep(.ant-pagination-item-active a) {
    color: #fff;
}

/* 表格边框样式 */
:deep(.ant-table) {
    /* border: 1px solid #f0f0f0; */
    /* border-radius: 4px; */
}

:deep(.ant-table-thead > tr > th) {
    border-bottom: 2px solid #f0f0f0;
}

:deep(.ant-table-tbody > tr > td) {
    border-bottom: 1px solid #f0f0f0;
}

/* 优化表格响应式 */
@media (max-width: 768px) {
    .eval-page {
        padding: 10px;
    }

    .game-table {
        font-size: 14px;
    }

    :deep(.category-tag) {
        font-size: 10px;
        padding: 1px 6px;
    }

    :deep(.ant-pagination) {
        margin-top: 10px;
    }
}
</style>