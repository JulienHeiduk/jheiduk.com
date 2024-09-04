---
title: "Polars Tutorial Part 1: Efficient Data Manipulation Compared to Pandas"
date: 2024-09-04
draft: false
summary: "Polars offers a faster and more memory-efficient alternative to Pandas for data manipulation tasks, particularly with large datasets, due to its use of parallel processing, lazy evaluation, and Arrow memory format, making it an ideal tool for data engineers and scientists seeking enhanced performance."
---

## Introduction

In the realm of data manipulation, **Pandas** has been the go-to library for Python developers and data scientists due to its versatility and power. However, with the increasing size of datasets and the need for speed, alternatives like **Polars** have emerged. This tutorial will introduce you to Polars, demonstrating its efficiency and showcasing benchmarks that contrast its performance with Pandas through simple data manipulation tasks.

## What is Polars?

Polars is a fast DataFrame library designed for parallel execution and efficient memory management. Unlike Pandas, which is single-threaded and can struggle with large datasets, Polars leverages lazy evaluation and parallel processing, making it suitable for high-performance data manipulation.

### Key Benefits of Polars:
- **Speed**: Built with performance in mind, Polars utilizes Rust's concurrency features.
- **Memory Efficiency**: Uses Arrow memory format, ensuring minimal memory overhead.
- **Lazy Evaluation**: Only computes the result when needed, allowing for optimizations.

## Why Data Scientists and Engineers Should Use Polars

As data sets continue to grow in size and complexity, data scientists and data engineers require tools that can efficiently manage and process these large volumes of data without compromising performance. Here are key reasons why Polars should be included in the toolkit of data professionals, especially in data engineering:

1. **Scalability**: Polars is designed to handle large datasets with ease. Its parallel execution capabilities allow for efficient processing, making it possible to work with datasets that are too large for Pandas to handle smoothly.

2. **Performance**: With Polars' optimized implementation using Rust, it provides much faster execution time versus Pandas, particularly for computationally intensive tasks. This speed can significantly decrease the time taken to derive insights and complete data engineering tasks.

3. **Memory Management**: Polars utilizes Arrow's in-memory columnar format, which allows it to execute operations in a cache-friendly manner, reducing memory overhead. This is especially critical in data engineering tasks involving large data processing pipelines where memory usage can become a bottleneck.

4. **Integration with Modern Data Tools**: Polars seamlessly integrates with various data sources and formats, making it a versatile option for modern data workflows that often involve cloud databases, big data technologies, or ETL processes.

5. **DataFrame API Paradigm**: Its API is designed to be user-friendly for those transitioning from Pandas, allowing data professionals to leverage their existing knowledge while taking advantage of Polars' enhanced performance characteristics.

The need for speed, efficiency, and the ability to handle large datasets without crashing or running out of memory means that Polars is a valuable asset in the arsenal of data scientists and engineers today.

## Getting Started with Polars

Before we dive into the benchmarks, ensure you have both Pandas and Polars installed. You can do this with pip:

```bash
pip install pandas polars
```

### Creating a Dataset

To compare performance, we first need to create a dataset. We'll generate a CSV file with random data.

#### Dataset Creation Script: `benchmark-part-1/0_create_dataset.py`
```python
import pandas as pd
import numpy as np

def create_large_dataset(size=10**6):
    data = {
        "id": np.random.randint(1, 10000, size),
        "value_1": np.random.randn(size),
        "value_2": np.random.randn(size),
        "category": np.random.choice(['A', 'B', 'C', 'D'], size),
    }
    return data

dataset_size = 10**6
data_pandas = create_large_dataset(dataset_size)
df_pandas = pd.DataFrame(data_pandas)

df_pandas.to_csv('test_dataset.csv', index=False)
```

Run this script to generate a dataset of 1,000,000 records, then we can proceed with the benchmarking.

## Benchmarking Polars vs Pandas

We'll benchmark three common data manipulation tasks: reading a CSV file, adding a new column, and filtering rows.

### 1. Benchmarking CSV Reading

#### Benchmark Import Script: `benchmark-part-1/1_benchmark_read.py`
```python
import pandas as pd
import polars as pl
import timeit

def read_csv_pandas():
    pd.read_csv('test_dataset.csv')

def read_csv_polars():
    pl.read_csv('test_dataset.csv')

time_pandas_csv = timeit.timeit(read_csv_pandas, number=10)
time_polars_csv = timeit.timeit(read_csv_polars, number=10)
print(f"CSV Read - Pandas: {time_pandas_csv:.4f} s, Polars: {time_polars_csv:.4f} s")
```

The first script is focused on the importation of a CSV file.

![polars_data_1.png](/polars_data_1.png)

The results indicate that Polars significantly outperforms Pandas in terms of speed. Specifically:

- Pandas took approximately 4.2099 seconds to import the CSV.
- Polars took only 0.3743 seconds, which is roughly 11 times faster than Pandas.

This highlights Polars' efficiency, particularly in handling CSV reading tasks, making it an attractive alternative for those working with large datasets or needing higher performance for data operations.

**Results**: Polars 1 - Pandas 0

### 2. Benchmarking Column Creation

#### Column Creation Benchmark Script: `benchmark-part-1/2_create_column_benchmark.py`
```python
import pandas as pd
import polars as pl
import timeit

df_pandas = pd.read_csv('test_dataset.csv')
df_polars = pl.read_csv('test_dataset.csv')

def add_column_pandas():
    df_pandas['new_column'] = df_pandas['value_1'] * df_pandas['value_2']

def add_column_polars():
    df_polars.with_columns((pl.col('value_1') * pl.col('value_2')).alias('new_column'))

time_pandas_add_column = timeit.timeit(add_column_pandas, number=10)
time_polars_add_column = timeit.timeit(add_column_polars, number=10)
print(f"Add Column - Pandas: {time_pandas_add_column:.4f} s, Polars: {time_polars_add_column:.4f} s")
```

The second script is focused on the creation of a new column.

![polars_data_2.png](/polars_data_2.png)

The results show:

- Pandas took 0.0456 seconds to add the new column.
- Polars took 0.1218 seconds to perform the same operation.

In this case, Pandas is faster at adding a new column compared to Polars. This difference may be due to the specific internal optimizations of Pandas for such operations. While Polars excels in many tasks, as seen with CSV reading, there are cases where Pandas can still outperform it in specific, simpler operations like adding columns.

**Results**: Polars 1 - Pandas 1

### 3. Benchmarking Row Filtering

#### Row Filtering Benchmark Script: `benchmark-part-1/3_filter_benchmark.py`
```python
import pandas as pd
import polars as pl
import timeit

df_pandas = pd.read_csv('test_dataset.csv')
df_polars = pl.read_csv('test_dataset.csv')

def filter_rows_pandas():
    df_pandas[df_pandas['category'] == 'A']

def filter_rows_polars():
    df_polars.filter(pl.col('category') == 'A')

time_pandas_filter = timeit.timeit(filter_rows_pandas, number=10)
time_polars_filter = timeit.timeit(filter_rows_polars, number=10)
print(f"Filter Rows - Pandas: {time_pandas_filter:.4f} s, Polars: {time_polars_filter:.4f} s")
```

In this third benchmark comparison shown in the screenshot, the task being measured is row filtering.

![polars_data_3.png](/polars_data_3.png)

The results show the following performance:

- Pandas took 0.8262 seconds to filter the rows.
- Polars took only 0.1768 seconds for the same operation.

In this case, Polars is significantly faster than Pandas, taking roughly 5 times less time to filter rows. This is another example of how Polars can be more efficient for certain data manipulation tasks, especially those involving larger datasets or more computational complexity, leveraging its parallel execution and optimized processing.

**Results**: Polars 2 - Pandas 1

### 4. Benchmarking GroupBy Operations

#### GroupBy Benchmark Script: `benchmark-part-1/4_groupby_benchmark.py`
```python
import pandas as pd
import polars as pl
import timeit

df_pandas = pd.read_csv('test_dataset.csv')
df_polars = pl.read_csv('test_dataset.csv')

def groupby_pandas():
    df_pandas.groupby('category').agg({'value_1': 'mean'})

def groupby_polars():
    df_polars.groupby('category').agg(pl.col('value_1').mean())

time_pandas_groupby = timeit.timeit(groupby_pandas, number=10)
time_polars_groupby = timeit.timeit(groupby_polars, number=10)
print(f"Group By - Pandas: {time_pandas_groupby:.4f} s, Polars: {time_polars_groupby:.4f} s")
```

In this fourth screenshot, the benchmark compares the time taken to perform a groupby operation using Pandas and Polars.

![polars_data_4.png](/polars_data_4.png)

The results are:

- Pandas took 0.4196 seconds to complete the groupby operation.
- Polars took 0.2184 seconds for the same task.

Here, Polars is nearly twice as fast as Pandas in performing the groupby operation. This demonstrates Polars' advantage in efficiently handling more computationally intensive tasks, such as grouping and aggregating data, thanks to its optimized design for performance.

**Results**: Polars 3 - Pandas 1

## Conclusion

Through this tutorial, you have seen how Polars not only offers a powerful alternative to Pandas for handling large datasets but also provides an opportunity for better performance. In every task — reading CSV files, adding columns, filtering, and grouping — Polars consistently outperformed Pandas, making it a robust choice for data manipulation.

In our benchmark Polars is the winner with 3 points vs 1 for Pandas. We are going to go further in the next article.

**Next Steps**
As you progress in your data science journey, consider incorporating Polars into your workflow for better efficiency, especially when dealing with large datasets. Let's talk about more advanced features in the next article.