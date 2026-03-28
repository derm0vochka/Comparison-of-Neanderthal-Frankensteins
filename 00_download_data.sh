#!/bin/bash

# Скачивание всех необходимых данных 
# Запуск: bash 00_download_data.sh

cd ~/nd_pipeline/data/

# tsv-файлы IBS.YRI
# Выполнить локально
# Вручную копируем все IBS.YRI.grch37.chrN.em.tsv с помощью scp -P 'порт подключения' IBS.YRI.grch37.chrN.em.tsv 'логин':~/nd_pipeline/data/raw

# Карта рекомбинации HapMap II (GRCh37/hg19)
cd ~/nd_pipeline/data/hapmap
wget ftp://ftp.ncbi.nlm.nih.gov/hapmap/recombination/2011-01_phaseII_B37/genetic_map_HapMapII_GRCh37.tar.gz

# GTEx v7 cis-eQTL
# Заходим на https://gtexportal.org, перевходим сверху в раздел Downloads
# Сверху ищем QTL, затем листаем вниз ищем GTEx Analysis V7, Single-Tissue cis-eQTL
# Локально нужно скачать GTEx_Analysis_v7_eQTL.tar.gz
# Вроде как не особо нужно GTEx_Analysis_v7_eQTL_expression_matrices.tar.gz, GTEx_Analysis_v7_eQTL_covariates.tar.gz, но можно тоже скачать
# Локально выполнить, чтобы скачать scp -P 'порт подключения' GTEx_Analysis_v7_eQTL.tar.gz 'логин':~/nd_pipeline/data/gtex
# Затем распакуем
wget https://storage.googleapis.com/adult-gtex/bulk-qtl/v7/single-tissue-cis-qtl/GTEx_Analysis_v7_eQTL.tar.gz
tar -xf GTEx_Analysis_v7_eQTL.tar

# Vindija 33.19 VCF (chr6, hg19)
cd ~/nd_pipeline/data/vindija
wget -c https://cdna.eva.mpg.de/neandertal/Vindija/VCF/Vindija33.19/chr6_mq25_mapab100.vcf.gz
wget -c https://cdna.eva.mpg.de/neandertal/Vindija/VCF/Vindija33.19/chr6_mq25_mapab100.vcf.gz.tbi

# GENCODE v19 (hg19) 
cd ~/nd_pipeline/data/gencode
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19/gencode.v19.annotation.gtf.gz
# Затем распакуем
gunzip -k gencode.v19.annotation.gtf.gz
