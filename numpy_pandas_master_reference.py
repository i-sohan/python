"""
NumPy & Pandas Complete Master Reference Guide
"""
import os
import tempfile

# ===========================================================================
# # NUMPY & PANDAS COMPLETE MASTER REFERENCE GUIDE
# This guide contains an exhaustive, production-grade reference of NumPy and Pandas functionality, covering array creation, data manipulation, indexing, linear algebra, time-series, aggregations, performance optimization, and file I/O with clear code examples.
# ===========================================================================\n
# ===========================================================================
# ## PART 1: NUMPY MASTER GUIDE
# NumPy (Numerical Python) is the foundational library for scientific computing in Python, offering N-dimensional array objects and high-performance vectorized operations.
# ===========================================================================\n
# ---------------------------------------------------------------------------
# ### 1. NumPy Array Creation & Initialization
# Demonstrates every method for initializing 1D, 2D, and multi-dimensional NumPy arrays.
# ---------------------------------------------------------------------------
import numpy as np

print("--- 1. NumPy Array Creation & Initialization ---")

# 1. From Python sequences
arr_1d = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# 2. Built-in constructors
zeros = np.zeros((3, 4))                  # Array of 0s
ones = np.ones((2, 3), dtype=np.int32)     # Array of 1s
full = np.full((3, 3), fill_value=7)       # Array with constant value
eye = np.eye(4)                            # 4x4 Identity matrix with 1s on diagonal
identity = np.identity(3)                  # 3x3 Square identity matrix
empty = np.empty((2, 2))                   # Uninitialized memory allocation

# 3. Numerical ranges and spacing
arange_arr = np.arange(0, 10, 2)           # Start, Stop (excl), Step -> [0, 2, 4, 6, 8]
linspace_arr = np.linspace(0, 1, 5)        # 5 evenly spaced values between 0 and 1
logspace_arr = np.logspace(1, 3, 3)        # Logarithmically spaced [10^1, 10^2, 10^3]
geomspace_arr = np.geomspace(1, 1000, 4)   # Geometrically spaced [1, 10, 100, 1000]

# 4. Coordinate grids
x = np.array([1, 2, 3])
y = np.array([4, 5])
xx, yy = np.meshgrid(x, y)                 # Generate 2D coordinate grid

# 5. Like-constructors (matches shape & dtype of existing array)
zeros_like = np.zeros_like(arr_2d)
ones_like = np.ones_like(arr_2d)
full_like = np.full_like(arr_2d, fill_value=99.9)

print(f"1D Array: {arr_1d}")
print(f"2D Float Array:\n{arr_2d}")
print(f"Linspace: {linspace_arr}")
print(f"Meshgrid XX:\n{xx}")

# ---------------------------------------------------------------------------
# ### 2. NumPy Data Types & Array Inspection
# Explores primitive data types, explicit type casting, and intrinsic array attributes.
# ---------------------------------------------------------------------------
print("--- 2. NumPy Data Types & Array Inspection ---")

# Standard primitive data types
arr_int = np.array([1, 2, 3], dtype=np.int32)
arr_float = np.array([1.5, 2.5, 3.5], dtype=np.float64)
arr_complex = np.array([1+2j, 3+4j], dtype=np.complex128)
arr_bool = np.array([True, False, True], dtype=np.bool_)
arr_str = np.array(["apple", "banana", "cherry"], dtype=np.str_)

# Type conversion using astype()
arr_converted = arr_float.astype(np.int64)

# Inspection attributes
sample = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int64)

print(f"Dimensions (.ndim): {sample.ndim}")
print(f"Shape (.shape): {sample.shape}")
print(f"Total Elements (.size): {sample.size}")
print(f"Data Type (.dtype): {sample.dtype}")
print(f"Element Size in Bytes (.itemsize): {sample.itemsize}")
print(f"Total Array Memory (.nbytes): {sample.nbytes} bytes")
print(f"Transposed Array Shape (.T.shape): {sample.T.shape}")
print(f"Memory Flags:\n{sample.flags}")

# ---------------------------------------------------------------------------
# ### 3. NumPy Indexing, Slicing & Subsetting
# Covers basic 1D/2D slicing, boolean masking, fancy integer indexing, dimension expansion, and ellipsis.
# ---------------------------------------------------------------------------
print("--- 3. NumPy Indexing, Slicing & Subsetting ---")

arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

# 1. Basic Slicing [start:stop:step]
print(f"1D Slice [2:7:2]: {arr[2:7:2]}")
print(f"Reversed 1D: {arr[::-1]}")
print(f"2D Sub-matrix [0:2, 1:3]:\n{matrix[0:2, 1:3]}")

# 2. Boolean Masking (Logical Indexing)
mask = (arr > 30) & (arr < 80)
print(f"Boolean Mask Filter (30 < x < 80): {arr[mask]}")

# 3. Fancy Indexing (Selecting specific rows/cols by list of indices)
row_indices = [0, 2]
col_indices = [1, 3]
print(f"Fancy Indexed Elements matrix[[0,2], [1,3]]: {matrix[row_indices, col_indices]}")
print(f"Fancy Indexed Rows matrix[[0, 2]]:\n{matrix[[0, 2]]}")

# 4. Dimension Expansion (np.newaxis & expand_dims)
vec_1d = np.array([1, 2, 3])
col_vec = vec_1d[:, np.newaxis]           # Shape (3, 1)
row_vec = np.expand_dims(vec_1d, axis=0)  # Shape (1, 3)
print(f"Column Vector Shape: {col_vec.shape}, Row Vector Shape: {row_vec.shape}")

# 5. Ellipsis (...) for multi-dimensional slicing
tensor = np.ones((2, 3, 4))
print(f"Ellipsis Slicing tensor[..., 0] Shape: {tensor[..., 0].shape}")

# ---------------------------------------------------------------------------
# ### 4. NumPy Shape Manipulation & Joining
# Reshaping, flattening, transposing, stacking, splitting, flipping, rolling, and expanding/squeezing arrays.
# ---------------------------------------------------------------------------
print("--- 4. NumPy Shape Manipulation & Joining ---")

arr = np.arange(12) # [0, 1, ..., 11]

# 1. Reshaping
grid = arr.reshape(3, 4)
auto_col = arr.reshape(2, -1)             # -1 automatically infers dimension size (2x6)

# 2. Flattening (ravel view vs flatten copy)
flattened_copy = grid.flatten()            # Returns a copy
raveled_view = grid.ravel()                # Returns a view where possible

# 3. Transposition & Axis Swap
transposed = grid.T                        # Shape (4, 3)
swapped = grid.swapaxes(0, 1)              # Swaps axis 0 and 1

# 4. Stacking and Concatenation
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

concat_row = np.concatenate((a, b), axis=0)  # Vertical join
concat_col = np.concatenate((a, b), axis=1)  # Horizontal join
vstack = np.vstack((a, b))                   # Vertical stack
hstack = np.hstack((a, b))                   # Horizontal stack
dstack = np.dstack((a, b))                   # Depth stack along 3rd axis
col_stack = np.column_stack(([1, 2], [3, 4]))# Stack 1D arrays as columns

# 5. Splitting
split_arrs = np.split(grid, 3, axis=0)       # Split into 3 equal sub-arrays
vsplit_arrs = np.vsplit(grid, 3)
hsplit_arrs = np.hsplit(grid, 2)

# 6. Rearranging Elements
flipped_v = np.flip(grid, axis=0)            # Flip vertically
rotated_90 = np.rot90(grid, k=1)             # Rotate 90 deg counter-clockwise
rolled = np.roll(arr, shift=3)               # Shift elements circularly

# 7. Squeezing (removing single-dimensional axes)
padded = np.zeros((1, 3, 1, 4))
squeezed = np.squeeze(padded)                # Shape becomes (3, 4)

print(f"Reshaped (3, 4):\n{grid}")
print(f"VStack Shape: {vstack.shape}")
print(f"HStack Shape: {hstack.shape}")
print(f"Squeezed Shape: {squeezed.shape}")

# ---------------------------------------------------------------------------
# ### 5. NumPy Math, Universal Functions (Ufuncs) & Broadcasting
# Demonstrates element-wise operations, vectorized mathematical ufuncs, statistics, aggregations, and broadcasting rules.
# ---------------------------------------------------------------------------
print("--- 5. NumPy Math, Ufuncs & Broadcasting ---")

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

# 1. Element-wise arithmetic
add_res = x + y
sub_res = y - x
mul_res = x * y
div_res = y / x
pow_res = x ** 2

# 2. Universal Functions (ufuncs)
sin_vals = np.sin(x)
exp_vals = np.exp(x)
log_vals = np.log(x)
sqrt_vals = np.sqrt(x)
clipped = np.clip(y, a_min=15, a_max=45)    # Clamps values to [15, 45]

# 3. Aggregations & Statistics across axes
data = np.array([[10, 20, 30],
                 [40, 50, 60]])

sum_total = np.sum(data)
sum_cols = np.sum(data, axis=0)             # Sum along rows for each column -> [50, 70, 90]
sum_rows = np.sum(data, axis=1)             # Sum along columns for each row -> [60, 150]
mean_val = np.mean(data)
std_val = np.std(data)
var_val = np.var(data)
min_idx = np.argmin(data)                   # Index of global minimum
max_idx = np.argmax(data)                   # Index of global maximum
cumsum_vals = np.cumsum(x)                  # Cumulative sum
p75 = np.percentile(data, 75)               # 75th percentile
q50 = np.quantile(data, 0.5)                # 50th quantile (median)

# 4. Broadcasting Rules Example
# Rule: Dimensions are compatible if they are equal or one of them is 1.
mat = np.ones((3, 3))
row_vec = np.array([1, 2, 3])
col_vec = np.array([[10], [20], [30]])

broadcast_row = mat + row_vec                # Adds [1, 2, 3] to each row
broadcast_col = mat + col_vec                # Adds [[10],[20],[30]] to each col
outer_broadcast = row_vec + col_vec          # (1, 3) + (3, 1) -> (3, 3) matrix

print(f"Clipped y: {clipped}")
print(f"Sum across cols (axis=0): {sum_cols}")
print(f"75th Percentile: {p75}")
print(f"Outer Broadcast (Row + Col Vector):\n{outer_broadcast}")

# ---------------------------------------------------------------------------
# ### 6. NumPy Random Number Generation (`np.random`)
# Modern `default_rng()` API for generating pseudo-random values across standard distributions, sampling, and shuffling.
# ---------------------------------------------------------------------------
print("--- 6. NumPy Random Number Generation ---")

# Initialize modern Generator instance with reproducible seed
rng = np.random.default_rng(seed=42)

# 1. Uniform & Normal distributions
rand_uniform = rng.random((2, 3))                   # Uniform float in [0.0, 1.0)
rand_range = rng.uniform(low=10.0, high=50.0, size=(2, 2))
std_normal = rng.standard_normal((3, 3))            # Standard normal distribution (mean=0, std=1)
custom_normal = rng.normal(loc=100, scale=15, size=5)# Mean=100, Std=15

# 2. Integers and Discrete distributions
rand_ints = rng.integers(low=1, high=100, size=(3, 3), endpoint=False)
poisson_draws = rng.poisson(lam=3.0, size=5)
binomial_draws = rng.binomial(n=10, p=0.5, size=5)

# 3. Sampling, Shuffling & Permutations
items = np.array(['A', 'B', 'C', 'D', 'E', 'F'])
sample = rng.choice(items, size=3, replace=False)   # Random choice without replacement
weighted_sample = rng.choice(items, size=3, p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1])

arr_to_shuffle = np.array([1, 2, 3, 4, 5])
rng.shuffle(arr_to_shuffle)                         # In-place shuffle
permuted = rng.permutation(items)                    # Returns new shuffled copy

print(f"Random Uniform Matrix:\n{rand_uniform}")
print(f"Random Integers:\n{rand_ints}")
print(f"Random Choice (no replace): {sample}")
print(f"In-place Shuffled Array: {arr_to_shuffle}")

# ---------------------------------------------------------------------------
# ### 7. NumPy Linear Algebra (`np.linalg`)
# Exhaustive linear algebra toolkit: matrix multiplication, inverse, determinant, rank, eigenvalues, SVD, QR, and solving linear systems.
# ---------------------------------------------------------------------------
print("--- 7. NumPy Linear Algebra ---")

A = np.array([[4.0, 2.0], [1.0, 3.0]])
B = np.array([[1.0, 5.0], [2.0, 0.0]])
v = np.array([1.0, 2.0])

# 1. Matrix Multiplication
matmul_operator = A @ B                             # Python matrix multiplication operator
matmul_func = np.matmul(A, B)                      # Matrix product of two arrays
dot_product = np.dot(A, B)                         # Dot product
outer_product = np.outer(v, v)                      # Outer product of two vectors
inner_product = np.inner(v, v)                      # Inner product of two vectors

# 2. Matrix Properties
det_A = np.linalg.det(A)                           # Determinant
trace_A = np.trace(A)                               # Sum of diagonal elements
rank_A = np.linalg.matrix_rank(A)                   # Matrix rank
norm_A = np.linalg.norm(A, ord='fro')               # Frobenius norm

# 3. Inversion & Pseudo-Inverse
inv_A = np.linalg.inv(A)                            # Multiplicative inverse of matrix
pinv_A = np.linalg.pinv(A)                          # Moore-Penrose pseudo-inverse

# 4. Decompositions & Eigen-problems
eigenvalues, eigenvectors = np.linalg.eig(A)        # Eigenvalues and right eigenvectors
U, S, Vt = np.linalg.svd(A)                        # Singular Value Decomposition
Q, R = np.linalg.qr(A)                              # QR Decomposition

# 5. Solving Linear Systems Ax = b
b = np.array([8.0, 13.0])
x_solution = np.linalg.solve(A, b)                  # Solves Ax = b exactly
lstsq_sol, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None) # Least-squares solution

print(f"Matrix A @ B:\n{matmul_operator}")
print(f"Determinant of A: {det_A:.4f}")
print(f"Inverse of A:\n{inv_A}")
print(f"Eigenvalues of A: {eigenvalues}")
print(f"Solution to Ax = b: {x_solution}")

# ---------------------------------------------------------------------------
# ### 8. NumPy Logic, Searching, Sorting & Set Operations
# Conditionals, sorting algorithms, positional lookup, element search, and set-theoretic operations.
# ---------------------------------------------------------------------------
print("--- 8. NumPy Logic, Searching, Sorting & Set Ops ---")

arr = np.array([12, 5, 8, 3, 19, 8, 1, 14])

# 1. Conditional Logic & Searching
where_res = np.where(arr > 10, 999, arr)             # Replace elements > 10 with 999
indices_gt_10 = np.argwhere(arr > 10)                # Indices where condition is True
nonzero_indices = np.nonzero(arr)                    # Indices of non-zero elements

sorted_arr = np.array([10, 20, 30, 40, 50])
insert_pos = np.searchsorted(sorted_arr, 25)        # Finds index to insert 25 to maintain order

# 2. Boolean Testing Functions
has_nan = np.isnan(np.array([1, 2, np.nan, 4]))
has_inf = np.isinf(np.array([1, np.inf, 3]))
all_positive = np.all(arr > 0)                       # Checks if ALL elements satisfy condition
any_gt_15 = np.any(arr > 15)                         # Checks if ANY element satisfies condition

# 3. Sorting Algorithms & Partitioning
sorted_copy = np.sort(arr)                           # Returns sorted copy
sort_order_indices = np.argsort(arr)                 # Returns indices that would sort array
partitioned = np.partition(arr, kth=3)               # Rearranges array so 3rd element is in sorted position

# 4. Set Operations
s1 = np.array([1, 2, 3, 4, 5, 5])
s2 = np.array([4, 5, 6, 7, 8])

unique_vals, counts = np.unique(s1, return_counts=True)
intersection = np.intersect1d(s1, s2)
union = np.union1d(s1, s2)
diff = np.setdiff1d(s1, s2)                          # In s1 but not in s2
in_s2_mask = np.isin(s1, s2)                         # Elementwise check if s1 elements exist in s2

print(f"np.where result: {where_res}")
print(f"Insertion index for 25 in sorted array: {insert_pos}")
print(f"Sorted array: {sorted_copy}")
print(f"Argsort indices: {sort_order_indices}")
print(f"Unique elements of s1: {unique_vals} (Counts: {counts})")
print(f"Intersection of s1 & s2: {intersection}")

# ---------------------------------------------------------------------------
# ### 9. Advanced NumPy Features & File I/O
# Structured arrays, masked arrays, sliding windows, binary file I/O (.npy/.npz), and text I/O.
# ---------------------------------------------------------------------------
import os
import tempfile

print("--- 9. Advanced NumPy Features & File I/O ---")

# 1. Structured / Record Arrays (C-style struct equivalent)
dtype_spec = [('name', 'U10'), ('age', 'i4'), ('salary', 'f8')]
employees = np.array([('Alice', 30, 85000.0), ('Bob', 35, 92000.0)], dtype=dtype_spec)
print(f"Structured Array Names: {employees['name']}")
print(f"Structured Array Salaries: {employees['salary']}")

# 2. Masked Arrays (handling invalid / missing values)
raw_data = np.array([10, -999, 20, 30, -999, 40])
masked_data = np.ma.masked_where(raw_data == -999, raw_data)
print(f"Masked Array Mean (ignoring -999): {masked_data.mean()}")

# 3. Sliding Window View (Stride Tricks)
arr_sequence = np.arange(10)
window_view = np.lib.stride_tricks.sliding_window_view(arr_sequence, window_shape=3)
print(f"Sliding Window (Size 3):\n{window_view}")

# 4. Binary & Text File I/O
with tempfile.TemporaryDirectory() as tmpdir:
    npy_path = os.path.join(tmpdir, "data.npy")
    npz_path = os.path.join(tmpdir, "archive.npz")
    txt_path = os.path.join(tmpdir, "data.txt")
    
    # Save & Load single array (.npy)
    np.save(npy_path, arr_sequence)
    loaded_npy = np.load(npy_path)
    
    # Save & Load multiple arrays compressed (.npz)
    np.savez_compressed(npz_path, a=arr_sequence, b=employees)
    with np.load(npz_path) as archive:
        loaded_a = archive['a']
        
    # Text I/O (.txt / .csv)
    np.savetxt(txt_path, np.array([[1.5, 2.5], [3.5, 4.5]]), delimiter=',')
    loaded_txt = np.loadtxt(txt_path, delimiter=',')

print("File I/O operations executed successfully.")

# ===========================================================================
# ## PART 2: PANDAS MASTER GUIDE
# Pandas provides high-performance, easy-to-use data structures (Series and DataFrame) and data analysis tools built on top of NumPy.
# ===========================================================================\n
# ---------------------------------------------------------------------------
# ### 1. Pandas Data Structures & Data Creation
# Creating Series and DataFrames from Python dicts, lists, NumPy arrays, and inspect basic structural metadata.
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np

print("--- 1. Pandas Data Structures & Data Creation ---")

# 1. pd.Series Creation
s1 = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'], name='scores', dtype=np.float64)
s_dict = pd.Series({'Apple': 1.5, 'Banana': 0.75, 'Cherry': 2.5})

# 2. pd.DataFrame Creation
# From Dict of Lists
df_from_dict = pd.DataFrame({
    'ID': [101, 102, 103, 104, 105],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [25, 30, 35, 40, 28],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'New York'],
    'Salary': [70000.0, 85000.0, 95000.0, 62000.0, 78000.0],
    'JoinDate': pd.date_range('2023-01-01', periods=5, freq='ME')
})

# From List of Dicts
df_from_records = pd.DataFrame.from_records([
    {'x': 1, 'y': 10},
    {'x': 2, 'y': 20}
])

# From 2D NumPy Array with Custom Columns
df_from_numpy = pd.DataFrame(
    np.random.default_rng(42).random((4, 3)),
    columns=['Metric_A', 'Metric_B', 'Metric_C']
)

print(f"Series:\n{s1}")
print(f"DataFrame Head:\n{df_from_dict.head(3)}")

# ---------------------------------------------------------------------------
# ### 2. Pandas Inspection & Data Profiling
# Methods for profiling shape, summary statistics, memory consumption, column types, and value distributions.
# ---------------------------------------------------------------------------
print("--- 2. Pandas Inspection & Data Profiling ---")

df = pd.DataFrame({
    'Department': ['HR', 'Tech', 'Tech', 'HR', 'Finance', 'Tech'],
    'Experience': [2, 5, 8, 1, 10, 4],
    'Score': [88.5, 92.0, 95.5, 76.0, 89.0, 91.0]
})

print(f"Shape (.shape): {df.shape}")
print(f"Columns (.columns): {list(df.columns)}")
print(f"Data Types (.dtypes):\n{df.dtypes}")
print(f"Memory Usage (.memory_usage):\n{df.memory_usage(deep=True)}")

print("\nDataFrame Info (.info()):")
df.info()

print("\nSummary Statistics (.describe()):")
print(df.describe(include='all'))

print("\nValue Counts (.value_counts()):")
print(df['Department'].value_counts(normalize=True))

# ---------------------------------------------------------------------------
# ### 3. Pandas Indexing, Selection & Filtering
# Exhaustive indexing tools: `.loc`, `.iloc`, `.at`, `.iat`, boolean masking, multi-condition filtering, and `.query()`.
# ---------------------------------------------------------------------------
print("--- 3. Pandas Indexing, Selection & Filtering ---")

df = pd.DataFrame({
    'EmpID': ['E01', 'E02', 'E03', 'E04', 'E05'],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [25, 32, 45, 29, 38],
    'Salary': [70000, 85000, 120000, 65000, 95000],
    'Dept': ['IT', 'HR', 'IT', 'Sales', 'HR']
}, index=['r1', 'r2', 'r3', 'r4', 'r5'])

# 1. Label-based Selection (.loc[row, col])
loc_sub = df.loc[['r1', 'r3'], ['Name', 'Salary']]

# 2. Integer-positional Selection (.iloc[row, col])
iloc_sub = df.iloc[0:3, 1:4]

# 3. Fast Scalar Access (.at and .iat)
scalar_label = df.at['r2', 'Salary']
scalar_pos = df.iat[1, 3]

# 4. Multi-Condition Boolean Filtering
bool_filter = df[(df['Age'] > 28) & (df['Dept'] == 'IT')]
isin_filter = df[df['Dept'].isin(['HR', 'Sales'])]
between_filter = df[df['Salary'].between(70000, 100000)]

# 5. Expression-based Query (.query())
query_res = df.query('Age >= 30 and Dept in ["IT", "HR"]')

# 6. Setting / Resetting Index
df_indexed = df.set_index('EmpID')
df_reset = df_indexed.reset_index(drop=False)

print(f"loc selection:\n{loc_sub}")
print(f"iloc selection:\n{iloc_sub}")
print(f"Boolean Filter (Age > 28 & Dept == IT):\n{bool_filter}")
print(f"Query Result:\n{query_res}")

# ---------------------------------------------------------------------------
# ### 4. Pandas Data Cleaning & Preprocessing
# Handling missing data (NaNs), removing duplicates, data type conversions, string manipulation (`.str`), and column operations.
# ---------------------------------------------------------------------------
print("--- 4. Pandas Data Cleaning & Preprocessing ---")

raw_df = pd.DataFrame({
    'raw_name': ['  alice smith ', 'BOB JONES', 'charlie brown', 'alice smith ', None],
    'age': ['25', '30', 'INVALID', '25', '40'],
    'score': [90.0, np.nan, 85.0, 90.0, np.nan],
    'phone': ['123-456-7890', '987-654-3210', '555-123-4567', '123-456-7890', None]
})

# 1. Handling Missing Data
isna_mask = raw_df.isna()
dropped_na = raw_df.dropna(subset=['raw_name', 'score'])
filled_const = raw_df['score'].fillna(0.0)
filled_ffill = raw_df['score'].ffill()                       # Forward fill
interpolated = raw_df['score'].interpolate(method='linear')  # Linear interpolation

# 2. Removing Duplicates
duplicates_mask = raw_df.duplicated(subset=['raw_name'])
deduplicated = raw_df.drop_duplicates(subset=['raw_name'], keep='first')

# 3. Robust Type Casting
numeric_age = pd.to_numeric(raw_df['age'], errors='coerce') # Converts INVALID to NaN

# 4. String Operations via .str Accessor
clean_names = raw_df['raw_name'].str.strip().str.title()
has_smith = raw_df['raw_name'].str.contains('smith', case=False, na=False)
split_names = clean_names.str.split(' ', expand=True)
split_names.columns = ['FirstName', 'LastName']

# 5. Renaming and Column Assignment
cleaned_df = (raw_df
    .assign(
        Name=clean_names,
        Age=numeric_age,
        Score=interpolated
    )
    .drop(columns=['raw_name', 'age'])
    .rename(columns={'phone': 'ContactNumber'})
)

print(f"Deduplicated DataFrame:\n{deduplicated}")
print(f"Extracted First/Last Names:\n{split_names}")
print(f"Cleaned DataFrame Pipeline Result:\n{cleaned_df}")

# ---------------------------------------------------------------------------
# ### 5. Pandas Data Transformation & Computations
# Applying functions (`map`, `apply`, `transform`), binning numerical data (`cut`, `qcut`), categorical encoding, and rolling indicators.
# ---------------------------------------------------------------------------
print("--- 5. Pandas Data Transformation & Computations ---")

df = pd.DataFrame({
    'Employee': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Salary': [50000, 75000, 120000, 45000, 95000],
    'BonusPct': [0.10, 0.15, 0.20, 0.05, 0.12],
    'Dept': ['Sales', 'IT', 'Exec', 'Sales', 'IT']
})

# 1. Map, Apply, and Transform
# Series.map (dict lookup or function)
dept_code = df['Dept'].map({'Sales': 'S', 'IT': 'T', 'Exec': 'E'})

# DataFrame.apply across axis=1 (row-wise)
df['TotalComp'] = df.apply(lambda row: row['Salary'] * (1 + row['BonusPct']), axis=1)

# Groupwise transform (broadcasts summary back to original shape)
df['DeptAvgSalary'] = df.groupby('Dept')['Salary'].transform('mean')

# 2. Binning & Quantile Discretization
df['SalaryTier'] = pd.cut(
    df['Salary'],
    bins=[0, 60000, 100000, np.inf],
    labels=['Entry', 'Mid', 'Senior']
)
df['IncomeQuartile'] = pd.qcut(df['Salary'], q=3, labels=['Q1', 'Q2', 'Q3'])

# 3. Categorical Data Type Optimization
df['Dept'] = df['Dept'].astype('category')
print(f"Category Codes: {df['Dept'].cat.codes.tolist()}")

# 4. Rank, Shift, and Percentage Change
df['SalaryRank'] = df['Salary'].rank(ascending=False, method='dense')
df['SalaryPctDiff'] = df['Salary'].pct_change()
df['SalaryDiff'] = df['Salary'].diff()

print(f"Transformed DataFrame:\n{df[['Employee', 'Salary', 'TotalComp', 'DeptAvgSalary', 'SalaryTier', 'SalaryRank']]}")

# ---------------------------------------------------------------------------
# ### 6. Pandas Grouping & Aggregations (`groupby`)
# Group-by operations, multi-column custom aggregations (`.agg()`), group filtering, and group iterative workflows.
# ---------------------------------------------------------------------------
print("--- 6. Pandas Grouping & Aggregations ---")

sales_df = pd.DataFrame({
    'Store': ['Store_A', 'Store_A', 'Store_A', 'Store_B', 'Store_B', 'Store_B'],
    'Category': ['Tech', 'Tech', 'Office', 'Tech', 'Office', 'Office'],
    'Sales': [250, 300, 150, 400, 100, 180],
    'Returns': [5, 10, 2, 8, 1, 4]
})

# 1. Basic GroupBy Aggregation
single_group = sales_df.groupby('Store')['Sales'].sum()

# 2. Multi-Column & Multi-Aggregation (.agg())
multi_agg = sales_df.groupby(['Store', 'Category']).agg(
    Total_Sales=('Sales', 'sum'),
    Avg_Sales=('Sales', 'mean'),
    Max_Returns=('Returns', 'max'),
    Sales_Std=('Sales', lambda x: x.std() if len(x) > 1 else 0.0)
)

# 3. Group Filtering (keeps groups meeting condition)
filtered_groups = sales_df.groupby('Store').filter(lambda g: g['Sales'].sum() > 600)

# 4. Iterating over Groups
print("Iterating over Store Groups:")
for store_name, group_frame in sales_df.groupby('Store'):
    print(f"Store: {store_name}, Total Items: {len(group_frame)}")

print(f"Single Group Sum:\n{single_group}")
print(f"Multi Aggregation Table:\n{multi_agg}")
print(f"Filtered Groups DataFrame:\n{filtered_groups}")

# ---------------------------------------------------------------------------
# ### 7. Pandas Merging, Joining & Concatenation
# Concatenating along axes, SQL-style merges (`inner`, `left`, `right`, `outer`, `cross`), and index joins.
# ---------------------------------------------------------------------------
print("--- 7. Pandas Merging, Joining & Concatenation ---")

df1 = pd.DataFrame({
    'KeyID': ['K0', 'K1', 'K2', 'K3'],
    'A': ['A0', 'A1', 'A2', 'A3'],
    'B': ['B0', 'B1', 'B2', 'B3']
})

df2 = pd.DataFrame({
    'KeyID': ['K1', 'K2', 'K3', 'K4'],
    'C': ['C1', 'C2', 'C3', 'C4'],
    'D': ['D1', 'D2', 'D3', 'D4']
})

# 1. Concatenation (pd.concat)
concat_rows = pd.concat([df1, df2], axis=0, ignore_index=True) # Stack vertically
concat_cols = pd.concat([df1.set_index('KeyID'), df2.set_index('KeyID')], axis=1) # Stack horizontally

# 2. Database Merging (pd.merge)
inner_merge = pd.merge(df1, df2, on='KeyID', how='inner')
left_merge = pd.merge(df1, df2, on='KeyID', how='left')
outer_merge = pd.merge(df1, df2, on='KeyID', how='outer', indicator=True)
cross_merge = pd.merge(df1[['KeyID', 'A']], df2[['KeyID', 'C']], how='cross')

# 3. Index Joining (.join)
left_df = df1.set_index('KeyID')
right_df = df2.set_index('KeyID')
joined_df = left_df.join(right_df, how='left')

# 4. Combine First & Update
s_base = pd.Series([1, np.nan, 3, np.nan])
s_fill = pd.Series([10, 20, 30, 40])
combined_series = s_base.combine_first(s_fill)

print(f"Inner Merge:\n{inner_merge}")
print(f"Outer Merge with Indicator:\n{outer_merge}")
print(f"Joined DataFrame:\n{joined_df}")

# ---------------------------------------------------------------------------
# ### 8. Pandas Reshaping, Pivoting & Crosstabs
# Pivot tables, unpivoting (`melt`), stacking/unstacking, cross-tabulations, and list exploding.
# ---------------------------------------------------------------------------
print("--- 8. Pandas Reshaping, Pivoting & Crosstabs ---")

data = pd.DataFrame({
    'Year': [2023, 2023, 2023, 2024, 2024, 2024],
    'Quarter': ['Q1', 'Q2', 'Q3', 'Q1', 'Q2', 'Q3'],
    'Region': ['North', 'North', 'South', 'North', 'South', 'South'],
    'Revenue': [100, 120, 90, 110, 130, 105],
    'Expenses': [60, 70, 50, 65, 75, 55]
})

# 1. Pivot Tables (pd.pivot_table)
pivot = pd.pivot_table(
    data,
    values='Revenue',
    index='Year',
    columns=['Region', 'Quarter'],
    aggfunc='sum',
    fill_value=0,
    margins=True                                     # Add Row/Col totals
)

# 2. Unpivoting / Melting (pd.melt)
melted = pd.melt(
    data,
    id_vars=['Year', 'Quarter', 'Region'],
    value_vars=['Revenue', 'Expenses'],
    var_name='Metric',
    value_name='Amount'
)

# 3. Stacking and Unstacking
stacked = data.set_index(['Year', 'Quarter', 'Region']).stack()
unstacked = stacked.unstack(level='Quarter')

# 4. Cross-Tabulation (pd.crosstab)
crosstab_res = pd.crosstab(data['Year'], data['Region'], values=data['Revenue'], aggfunc='mean')

# 5. Exploding List Columns (.explode())
df_lists = pd.DataFrame({
    'User': ['Alice', 'Bob'],
    'Tags': [['python', 'pandas'], ['numpy', 'scipy', 'ml']]
})
exploded = df_lists.explode('Tags')

print(f"Pivot Table:\n{pivot}")
print(f"Melted Head:\n{melted.head(4)}")
print(f"Exploded List DataFrame:\n{exploded}")

# ---------------------------------------------------------------------------
# ### 9. Pandas Time Series Analysis
# Date ranges, datetime properties (`.dt`), resampling, rolling/expanding windows, shifting, and timezones.
# ---------------------------------------------------------------------------
print("--- 9. Pandas Time Series Analysis ---")

# 1. Datetime Range Generation
date_index = pd.date_range(start='2026-01-01', periods=100, freq='D')
ts_data = pd.DataFrame({
    'Sales': np.random.default_rng(42).integers(100, 500, size=100)
}, index=date_index)

# 2. Datetime Properties via .dt Accessor
ts_data['Year'] = ts_data.index.year
ts_data['Month'] = ts_data.index.month
ts_data['DayName'] = ts_data.index.day_name()
ts_data['IsWeekend'] = ts_data.index.dayofweek >= 5

# 3. Resampling (Downsampling / Upsampling)
monthly_sales = ts_data['Sales'].resample('ME').sum()
weekly_ohlc = ts_data['Sales'].resample('W').ohlc()   # Open-High-Low-Close summary

# 4. Rolling & Expanding Windows
ts_data['7D_MovingAvg'] = ts_data['Sales'].rolling(window=7, min_periods=1).mean()
ts_data['Expanding_Max'] = ts_data['Sales'].expanding().max()
ts_data['EWM_Mean'] = ts_data['Sales'].ewm(span=14).mean()

# 5. Shifting & Lags
ts_data['Sales_Lag1'] = ts_data['Sales'].shift(1)    # Yesterday's sales
ts_data['Sales_Diff'] = ts_data['Sales'].diff(1)     # Day-over-day difference

# 6. Timezone Handling
ts_utc = ts_data.index.tz_localize('UTC')
ts_ny = ts_utc.tz_convert('America/New_York')

print(f"Monthly Resampled Sales:\n{monthly_sales}")
print(f"Weekly OHLC Summary:\n{weekly_ohlc.head(3)}")
print(f"Time Series Rolling Statistics:\n{ts_data[['Sales', '7D_MovingAvg', 'Sales_Lag1', 'Sales_Diff']].head(5)}")

# ---------------------------------------------------------------------------
# ### 10. Pandas MultiIndex / Hierarchical Indexing
# Constructing multi-level indices, slicing with `.xs` and `IndexSlice`, and reordering levels.
# ---------------------------------------------------------------------------
print("--- 10. Pandas MultiIndex ---")

# 1. MultiIndex Construction
arrays = [
    ['North', 'North', 'South', 'South'],
    ['Store_1', 'Store_2', 'Store_1', 'Store_2']
]
multi_idx = pd.MultiIndex.from_arrays(arrays, names=['Region', 'Store'])

multi_df = pd.DataFrame(
    np.random.default_rng(42).integers(10, 100, size=(4, 2)),
    index=multi_idx,
    columns=['Q1_Sales', 'Q2_Sales']
)

# 2. MultiIndex Selection
north_sub = multi_df.loc['North']                     # Select top-level
store1_cross = multi_df.xs('Store_1', level='Store')  # Cross-section across sub-level

# Using IndexSlice
idx = pd.IndexSlice
slice_res = multi_df.loc[idx[:, 'Store_2'], 'Q1_Sales']

# 3. Rearranging Index Levels
swapped_levels = multi_df.swaplevel('Region', 'Store')
sorted_multi = multi_df.sort_index(level='Region')

print(f"MultiIndex DataFrame:\n{multi_df}")
print(f"Cross Section for Store_1:\n{store1_cross}")
print(f"Swapped Levels DataFrame:\n{swapped_levels}")

# ---------------------------------------------------------------------------
# ### 11. Pandas Data Import & Export (I/O Operations)
# Reading and writing CSV, Excel, JSON, Parquet, and SQL formats.
# ---------------------------------------------------------------------------
import os
import tempfile

print("--- 11. Pandas Data Import & Export ---")

df_sample = pd.DataFrame({
    'A': [1, 2, 3],
    'B': ['x', 'y', 'z'],
    'C': [10.5, 20.5, 30.5]
})

with tempfile.TemporaryDirectory() as tmpdir:
    csv_file = os.path.join(tmpdir, "data.csv")
    json_file = os.path.join(tmpdir, "data.json")
    parquet_file = os.path.join(tmpdir, "data.parquet")
    
    # 1. CSV I/O
    df_sample.to_csv(csv_file, index=False)
    read_csv_df = pd.read_csv(csv_file)
    
    # 2. JSON I/O
    df_sample.to_json(json_file, orient='records')
    read_json_df = pd.read_json(json_file, orient='records')
    
    # 3. Parquet I/O (High performance columnar binary format)
    df_sample.to_parquet(parquet_file)
    read_parquet_df = pd.read_parquet(parquet_file)

print("CSV, JSON, and Parquet read/write routines verified successfully.")

# ---------------------------------------------------------------------------
# ### 12. Pandas Performance Optimization & Best Practices
# Memory downcasting, vectorization vs `.apply()` benchmarking, and clean method chaining with `.pipe()`.
# ---------------------------------------------------------------------------
import time

print("--- 12. Performance Optimization & Best Practices ---")

# 1. Memory Optimization via Type Downcasting
large_df = pd.DataFrame({
    'int_col': np.random.randint(0, 100, size=10000),
    'float_col': np.random.randn(10000),
    'category_col': np.random.choice(['GroupA', 'GroupB', 'GroupC'], size=10000)
})

mem_before = large_df.memory_usage(deep=True).sum()
large_df['int_col'] = pd.to_numeric(large_df['int_col'], downcast='integer')
large_df['float_col'] = pd.to_numeric(large_df['float_col'], downcast='float')
large_df['category_col'] = large_df['category_col'].astype('category')
mem_after = large_df.memory_usage(deep=True).sum()

print(f"Memory Reduction: {mem_before} bytes -> {mem_after} bytes ({(1 - mem_after/mem_before)*100:.2f}% saved)")

# 2. Vectorization vs Apply Speed Test
n_rows = 100000
test_df = pd.DataFrame({
    'a': np.random.rand(n_rows),
    'b': np.random.rand(n_rows)
})

# Iterative / Apply Approach
start = time.time()
res_apply = test_df.apply(lambda row: row['a'] + row['b'], axis=1)
time_apply = time.time() - start

# Vectorized Approach
start = time.time()
res_vectorized = test_df['a'] + test_df['b']
time_vectorized = time.time() - start

print(f"Apply Execution Time: {time_apply:.4f}s")
print(f"Vectorized Execution Time: {time_vectorized:.4f}s")
print(f"Vectorization Speedup: {time_apply / time_vectorized:.1f}x faster!")

# 3. Method Chaining Pipeline Pattern with .pipe()
def remove_outliers(df, col):
    q_low = df[col].quantile(0.01)
    q_high = df[col].quantile(0.99)
    return df[(df[col] >= q_low) & (df[col] <= q_high)]

def normalize_cols(df, cols):
    df_copy = df.copy()
    for col in cols:
        df_copy[col] = (df_copy[col] - df_copy[col].mean()) / df_copy[col].std()
    return df_copy

processed_df = (
    test_df
    .pipe(remove_outliers, col='a')
    .pipe(normalize_cols, cols=['a', 'b'])
    .assign(c_sum=lambda d: d['a'] + d['b'])
)

print(f"Pipeline Result Shape: {processed_df.shape}")
print("\nMaster NumPy and Pandas Reference execution completed!")
