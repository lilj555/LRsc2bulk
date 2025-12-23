# KEGG人类通路和基因信息获取脚本
# 使用KEGGREST包获取hsa（人类）的所有通路及其基因

library(KEGGREST)
library(dplyr)

# 设置输出目录
output_dir <- "/home/lilj/work/xenium/data"

# 获取人类所有通路
cat("正在获取人类所有通路...\n")
pathways <- keggList("pathway", "hsa")

# 创建数据框存储结果
pathway_gene_df <- data.frame(
  pathway_id = character(),
  pathway_name = character(),
  gene_id = character(),
  gene_symbol = character(),
  stringsAsFactors = FALSE
)

# 获取每个通路的基因信息
total_pathways <- length(pathways)
cat(paste("共找到", total_pathways, "个通路，开始获取基因信息...\n"))

for (i in 1:length(pathways)) {
  pathway_id <- names(pathways)[i]
  pathway_name <- pathways[i]
  
  # 显示进度
  if (i %% 10 == 0 || i == 1) {
    cat(paste("处理进度:", i, "/", total_pathways, "- 当前通路:", pathway_id, "\n"))
  }
  
  # 获取通路中的基因
  tryCatch({
    genes <- keggGet(pathway_id)[[1]]$GENE
    
    if (!is.null(genes)) {
      # 解析基因信息
      gene_entries <- genes[seq(1, length(genes), 2)]  # 奇数位置是基因ID
      gene_names <- genes[seq(2, length(genes), 2)]    # 偶数位置是基因名称
      
      # 提取基因符号（通常在分号前）
      gene_symbols <- sapply(gene_names, function(x) {
        parts <- strsplit(x, ";")[[1]]
        return(trimws(parts[1]))
      })
      
      # 创建当前通路的数据框
      current_df <- data.frame(
        pathway_id = rep(pathway_id, length(gene_entries)),
        pathway_name = rep(pathway_name, length(gene_entries)),
        gene_id = gene_entries,
        gene_symbol = gene_symbols,
        stringsAsFactors = FALSE
      )
      
      # 合并到主数据框
      pathway_gene_df <- rbind(pathway_gene_df, current_df)
    }
  }, error = function(e) {
    cat(paste("警告: 无法获取通路", pathway_id, "的基因信息:", e$message, "\n"))
  })
  
  # 添加小延迟避免过于频繁的API调用
  Sys.sleep(0.1)
}

# 输出结果统计
cat(paste("数据获取完成！\n"))
cat(paste("总通路数:", length(unique(pathway_gene_df$pathway_id)), "\n"))
cat(paste("总基因数:", length(unique(pathway_gene_df$gene_id)), "\n"))
cat(paste("总记录数:", nrow(pathway_gene_df), "\n"))

# 保存为CSV文件
output_file <- file.path(output_dir, "pathway_gene.csv")
write.csv(pathway_gene_df, output_file, row.names = FALSE)
cat(paste("结果已保存到:", output_file, "\n"))

# 显示前几行数据作为预览
cat("\n数据预览:\n")
print(head(pathway_gene_df, 10))