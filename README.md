# 值得苦练的Python经典题目

----

## 1.求最大公约数和最小公倍数

难度：<span style="background-color: #1E5D85; color: white; font-family: Arial, sans-serif; font-size: 14px; font-weight: bold; padding: 3px 10px; border-radius: 8px;">容易</span>

> 两个数的最大公约数是两个数的公共因子中最大的那个数；两个数的最小公倍数则是能够同时被两个数整除的最小的那个数。

函数签名：

```python
def gcd_lcm(num1: int, num2: int) -> tuple:
    pass
```

输入：

- num1：整数
- num2：整数

输出：

- 返回一个包含两个元素的元组，第一个元素表示最大公约数，第二个元素表示最小公倍数。

要求：

- 如果输入的参数不是正整数，函数应抛出异常。
- 返回的结果应为非负整数。

示例代码：

```python
def gcd_lcm(num1: int, num2: int) -> tuple:
    # 检查输入参数是否为正整数
    if not isinstance(num1, int) or not isinstance(num2, int) or num1 <= 0 or num2 <= 0:
        raise ValueError("参数必须为正整数")
    
    # 计算最大公约数
    def compute_gcd(a, b):
        while b != 0:
            a, b = b, a % b
        return a
    
    # 计算最小公倍数
    def compute_lcm(a, b):
        return abs(a * b) // compute_gcd(a, b)
    
    # 返回结果
    return (compute_gcd(num1, num2), compute_lcm(num1, num2))


# 测试示例
print(gcd_lcm(12, 18))
# 输出：(6, 36)

print(gcd_lcm(15, 20))
# 输出：(5, 60)
```

说明：

- 在`gcd_lcm`函数中，首先检查输入参数`num1`和`num2`是否为正整数，如果不是则抛出异常。
- 然后使用辗转相除法计算最大公约数，将计算公约数的逻辑封装在`compute_gcd`函数中。
- 最后通过计算最大公约数和输入参数的乘积除以最大公约数得到最小公倍数，将计算最小公倍数的逻辑封装在`compute_lcm`函数中。
- 返回一个包含最大公约数和最小公倍数的元组。



----

## 2.斐波那契数列

难度：<span style="background-color: #1E5D85; color: white; font-family: Arial, sans-serif; font-size: 14px; font-weight: bold; padding: 3px 10px; border-radius: 8px;">容易</span>

> 斐波那契数列（Fibonacci sequence），又称黄金分割数列，指的是这样一个数列：1、1、2、3、5、8、13、21、34、……。
>
> 在数学上，斐波那契数列以递归的方法来定义：
>
> $$
> \begin{equation}
> \left\{
> 	\begin{array}{ll}
> 		F(0)=0 \\
> 		F(1)=1 && (\boldsymbol{n} \in N) \\
> 		F(\boldsymbol{n})=F(\boldsymbol{n}-1)+F(\boldsymbol{n}+1)
> 	\end{array}\right.
> \end{equation}
> $$

函数签名：

```python
def fibonacci(n: int) -> int:
    pass
```

输入：

- num：非负整数，表示要返回的斐波那契数列的索引位置。

输出：

- 返回第n个斐波那契数列的值。

要求：

- 如果输入的参数不是非负整数，函数应抛出异常。

示例代码：

```python
def fibonacci(n: int) -> int:
    # 检查输入参数是否为非负整数
    if not isinstance(n, int) or n < 0:
        raise ValueError("参数必须为非负整数")

    # 计算斐波那契数列
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b


# 测试示例
print(fibonacci(0))
# 输出：0

print(fibonacci(1))
# 输出：1

print(fibonacci(10))
# 输出：55
```

说明：

- 在`fibonacci`函数中，首先检查输入参数`n`是否为非负整数，如果不是则抛出异常。
- 使用迭代的方式计算第n个斐波那契数列的值。使用两个变量`a`和`b`分别表示前一个斐波那契数列的值和当前斐波那契数列的值。
- 如果n为0，则返回0；如果n为1，则返回1；否则，通过迭代计算得到第n个斐波那契数列的值。
- 返回第n个斐波那契数列的值。

----

## 3.求前n阶乘的和

难度：<span style="background-color: #1E5D85; color: white; font-family: Arial, sans-serif; font-size: 14px; font-weight: bold; padding: 3px 10px; border-radius: 8px;">容易</span>

>比如求1+2!+3!+...+20!的和。

函数签名：

```python
def factorial_sum(n: int) -> int:
    pass
```

输入：

- num：正整数，表示要求前n阶乘的和。	<font color="red">注意：</font>不要输入太大的的n，防止数据过大造成电脑卡死。

输出：

- 返回前n阶乘的和。

要求：

- 如果输入的参数不是正整数，函数应抛出异常。

示例代码：

```python
def factorial_sum(num: int) -> int:
    # 检查输入参数是否为正整数
    if not isinstance(num, int) or num <= 0:
        raise ValueError("参数必须为正整数")

    # 计算前n阶乘的和
    a, b = 0, 1
    for i in range(1, n + 1):
        b *= i
        a += b
    
    return a


# 测试示例
print(factorial_sum(1))
# 输出：1

print(factorial_sum(2))
# 输出：3 (1! + 2!)

print(factorial_sum(5))
# 输出：153 (1! + 2! + 3! + 4! + 5!)
```

说明：

- 在`factorial_sum`函数中，首先检查输入参数`num`是否为正整数，如果不是则抛出异常。
- 使用循环计算前`num`阶乘的和。使用变量`a`来保存结果，初始值为0，变量`b`用于计算阶乘的值，初始值为1。在每次迭代中，`b`表示当前阶乘的值，通过乘以`i`来更新；`a`表示前`num`个阶乘的和，通过累加`b`到`a`中来更新。
- 返回前`num`阶乘的和。

----

## 4.判断年份是否是闰年

难度：<span style="background-color: #1E5D85; color: white; font-family: Arial, sans-serif; font-size: 14px; font-weight: bold; padding: 3px 10px; border-radius: 8px;">容易</span>

>判断该年年份是闰年的条件：
>
>- **能被4整除，但不能被100整除**
>- **能被400整除**

函数签名：

```python
def is_leap_year(year: int) -> bool:
    pass
```

输入：

- year：整数，表示要判断的年份。

输出：

- 返回一个布尔值，表示给定年份是否为闰年。如果是闰年，则返回`True`；否则返回`False`。

要求：

- 如果输入的参数不是整数，函数应抛出异常。

示例代码：

```python
def is_leap_year(year: int) -> bool:
    # 检查输入参数是否为整数
    if not isinstance(year, int):
        raise ValueError("参数必须为整数")

    # 判断是否为闰年
    if year % 400 == 0:
        return True
    elif year % 100 == 0:
        return False
    elif year % 4 == 0:
        return True
    else:
        return False


# 测试示例
print(is_leap_year(2000))
# 输出：True

print(is_leap_year(2020))
# 输出：True

print(is_leap_year(1900))
# 输出：False

print(is_leap_year(2022))
# 输出：False

```

说明：

- 在`is_leap_year`函数中，首先检查输入参数`year`是否为整数，如果不是则抛出异常。
- 使用条件判断来判断给定年份是否为闰年。根据闰年的定义，满足以下条件之一即可：
  - 能被400整除；
  - 能被4整除但不能被100整除。
- 根据条件依次判断，并返回相应的布尔值。如果满足条件，则返回`True`；否则返回`False`。

----

## 5.因式分解

难度：<span style="background-color: #EBA119; color: white; font-family: Arial, sans-serif; font-size: 14px; font-weight: bold; padding: 3px 10px; border-radius: 8px;">中等</span>

> 因式分解是将一个数表示为几个因子的乘积的过程，例如：$10=2\times5，60=2\times2\times3\times5$等。

函数签名：

```python
def factorize(num: int) -> str:
    pass
```

输入：

- num：正整数，要进行因式分解的数，要求小于1000。

输出：

- 返回一个字符串，表示因式分解的结果。

要求：

- 如果输入的参数不是正整数或大于等于1000，函数应抛出异常。
- 结果中的因子按照从小到大的顺序排列，并以`*`符号连接。

示例代码：

```python
def factorize(num: int) -> str:
    # 检查输入参数是否为正整数，并且小于1000
    if not isinstance(num, int) or num <= 0 or num >= 1000:
        raise ValueError("参数必须为正整数且小于1000")
    
    # 因式分解
    factors = []
    divisor = 2
    while num > 1:
        if num % divisor == 0:
            factors.append(str(divisor))
            num /= divisor
        else:
            divisor += 1
    
    # 返回结果
    return '*'.join(factors)


# 测试示例
print(factorize(12))
# 输出：2*2*3

print(factorize(36))
# 输出：2*2*3*3

print(factorize(90))
# 输出：2*3*3*5
```

说明：

- 在`factorize`函数中，首先检查输入参数`num`是否为正整数且小于1000，如果不满足要求则抛出异常。
- 使用一个循环来进行因式分解，每次找到一个能整除`num`的最小素数，并将其作为因子添加到结果列表中。然后将`num`除以该因子，继续寻找下一个因子，直到`num`等于1为止。
- 最后使用`*`符号连接结果列表中的因子，并返回一个字符串。

----

## 6.无重复字符的终止子串

难度：<span style="background-color: #EBA119; color: white; font-family: Arial, sans-serif; font-size: 14px; font-weight: bold; padding: 3px 10px; border-radius: 8px;">中等</span>

> 给定一个字符串`s`，请你找出其中不重复字符的 **最长子串** 的长度。

函数签名：

```python
def length_of_substring(s: str) -> int:
    pass
```

输入：

- s：字符串，表示输入的字符串。

输出：

- 返回一个整数，表示无重复字符的终止子串的长度。

要求：

- 字符串中只包含英文字母、数字和符号。
- 终止子串是指从字符串的某个位置开始，到最后一个不重复字符为止的子串。
- 考虑大小写是否敏感。

示例代码：

```python
def length_of_substring(s: str) -> int:
    # 检查输入参数是否合法
    if not isinstance(s, str):
        raise TypeError("输入必须为字符串")
    
    # 使用滑动窗口解决问题
    window = set()
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        # 如果右指针指向的字符在窗口中存在，则移动左指针直到窗口中不再存在该字符
        while s[right] in window:
            window.remove(s[left])
            left += 1
        
        # 将当前字符加入窗口中
        window.add(s[right])
        
        # 更新最大长度
        max_length = max(max_length, right - left + 1)
    
    # 返回结果
    return max_length


# 测试示例
print(length_of_substring("abcabcbb"))
# 输出：3

print(length_of_substring("bbbbb"))
# 输出：1

print(length_of_substring("pwwkew"))
# 输出：3

```

说明：

- 在`length_of_substring`函数中，首先检查输入参数`s`是否为字符串，如果不满足要求则抛出异常。
- 使用滑动窗口的思想来解决该问题。维护一个窗口，窗口内的字符是不重复的。
- 使用两个指针`left`和`right`分别表示窗口的左边界和右边界。
- 遍历字符串中的每个字符，如果当前字符已经在窗口中存在，则将左指针向右移动，直到窗口中不再存在该字符。
- 将当前字符加入窗口中，并更新最大长度。
- 最后返回最大长度。

----

## 7.组合总和

难度：<span style="background-color: #EBA119; color: white; font-family: Arial, sans-serif; font-size: 14px; font-weight: bold; padding: 3px 10px; border-radius: 8px;">中等</span>

> 给定一个无重复元素的正整数数组 `candidates` 和一个目标整数 `target`，找出数组中所有可以使数字和为 `target` 的不同组合。

函数签名：

```python
from typing import List

def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    pass
```

输入：

- candidates：一个无重复元素的正整数数组。
- target：目标整数。

输出：

- 返回一个列表，包含所有可以使数字和为 `target` 的不同组合。每个组合是一个列表，其中的数字按非递减顺序排列。

要求：

- 如果输入参数不符合要求，函数应抛出异常。

示例代码：

```python
from typing import List

def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    # 检查输入参数是否符合要求
    if not isinstance(candidates, list) or not all(isinstance(num, int) and num > 0 for num in candidates):
        raise ValueError("candidates必须是由正整数组成的列表")
    if not isinstance(target, int) or target <= 0:
        raise ValueError("target必须是正整数")

    # 定义结果列表
    result = []
    
    # 回溯算法搜索组合
    def backtrack(combination, start, target):
        if target == 0:
            result.append(combination)
            return
        for i in range(start, len(candidates)):
            if target < candidates[i]:
                break
            backtrack(combination + [candidates[i]], i, target - candidates[i])
    
    # 排序数组，并调用回溯算法
    candidates.sort()
    backtrack([], 0, target)
    
    return result


# 测试示例
print(combination_sum([2, 3, 6, 7], 7))
# 输出：[[2, 2, 3], [7]]

print(combination_sum([2, 3, 5], 8))
# 输出：[[2, 2, 2, 2], [2, 3, 3], [3, 5]]
```

说明：

- 在`combination_sum`函数中，首先检查输入参数`candidates`和`target`是否符合要求，如果不是则抛出异常。
- 定义一个空列表`result`用于存储结果。
- 使用回溯算法搜索所有可能的组合，将搜索过程封装在`backtrack`函数中。函数接受三个参数：当前组合`combination`、搜索起始位置`start`和剩余目标值`target`。当目标值等于0时，表示找到了一个满足条件的组合，将其添加到结果列表中。否则，遍历候选数字，并逐个尝试添加到组合中，然后继续递归搜索剩余部分。注意，在递归中更新搜索起始位置为当前位置，以确保不重复使用相同的数字。
- 在调用回溯算法之前，对候选数组进行排序，以便于去除重复的组合，并使组合中的数字按非递减顺序排列。
- 返回结果列表。

----

## 8.全排列

难度：<span style="background-color: #EBA119; color: white; font-family: Arial, sans-serif; font-size: 14px; font-weight: bold; padding: 3px 10px; border-radius: 8px;">中等</span>

>给定一个不含重复数字的数组 `nums`，返回其所有可能的全排列。

函数签名：

```python
from typing import List

def permute(nums: List[int]) -> List[List[int]]:
    pass
```

输入：

- nums：一个不含重复数字的整数数组。

输出：

- 返回一个列表，包含所有可能的全排列。每个排列是一个列表，其中的数字按任意顺序排列。

要求：

- 如果输入参数不符合要求，函数应抛出异常。

示例代码：

```python
from typing import List

def permute(nums: List[int]) -> List[List[int]]:
    # 检查输入参数是否符合要求
    if not isinstance(nums, list) or not all(isinstance(num, int) for num in nums):
        raise ValueError("nums必须是由整数组成的列表")

    # 定义结果列表
    result = []
    
    # 回溯算法生成全排列
    def backtrack(start):
        if start == len(nums):
            result.append(nums[:])
        for i in range(start, len(nums)):
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]
    
    # 调用回溯算法
    backtrack(0)
    
    return result


# 测试示例
print(permute([1, 2, 3]))
# 输出：[[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 2, 1], [3, 1, 2]]
```

说明：

- 在`permute`函数中，首先检查输入参数`nums`是否符合要求，如果不是则抛出异常。
- 定义一个空列表`result`用于存储结果。
- 使用回溯算法生成所有可能的全排列，将搜索过程封装在`backtrack`函数中。函数接受一个参数`start`，表示当前要进行交换的位置。当`start`等于数组长度时，表示已经生成了一个完整的排列，将其添加到结果列表中。否则，遍历数组中从`start`位置开始的每个数字，并将其与当前位置交换，然后继续递归生成剩余部分的排列。注意，在递归中恢复交换前的状态，以确保不会对后续的排列造成影响。
- 调用回溯算法，初始位置为0。
- 返回结果列表。

----

## 9.寻找两个正序数组的中位数

难度：<span style="background-color: #D62E04; color: white; font-family: Arial, sans-serif; font-size: 14px; font-weight: bold; padding: 3px 10px; border-radius: 8px;">困难</span>

> 给定两个大小分别为 `m` 和 `n` 的正序（从小到大）数组 `nums1` 和 `nums2`，请你找出并返回这两个正序数组的中位数。

函数签名：

```python
from typing import List

def find_median_sorted_arrays(nums1: List[int], nums2: List[int]) -> float:
    pass
```

输入：

- nums1：一个正序的整数数组。
- nums2：一个正序的整数数组。

输出：

- 返回两个正序数组的中位数。

要求：

- 如果输入参数不符合要求，函数应抛出异常。
- 算法的时间复杂度应为 `O(log (m+n))`。

示例代码：

```python
from typing import List

def find_median_sorted_arrays(nums1: List[int], nums2: List[int]) -> float:
    # 检查输入参数是否符合要求
    if not isinstance(nums1, list) or not all(isinstance(num, int) for num in nums1):
        raise ValueError("nums1必须是由整数组成的列表")
    if not isinstance(nums2, list) or not all(isinstance(num, int) for num in nums2):
        raise ValueError("nums2必须是由整数组成的列表")

    # 合并两个有序数组
    merged = []
    i, j = 0, 0
    while i < len(nums1) and j < len(nums2):
        if nums1[i] <= nums2[j]:
            merged.append(nums1[i])
            i += 1
        else:
            merged.append(nums2[j])
            j += 1
    while i < len(nums1):
        merged.append(nums1[i])
        i += 1
    while j < len(nums2):
        merged.append(nums2[j])
        j += 1
    
    # 计算中位数
    mid = len(merged) // 2
    if len(merged) % 2 == 0:
        return (merged[mid - 1] + merged[mid]) / 2
    else:
        return merged[mid]


# 测试示例
print(find_median_sorted_arrays([1, 3], [2]))
# 输出：2.0

print(find_median_sorted_arrays([1, 2], [3, 4]))
# 输出：2.5
```

说明：

- 在`find_median_sorted_arrays`函数中，首先检查输入参数`nums1`和`nums2`是否符合要求，如果不是则抛出异常。
- 合并两个有序数组，将其存储在一个新的列表`merged`中。使用双指针方法分别遍历`nums1`和`nums2`，比较指针位置上的元素大小，并将较小的元素添加到`merged`中。当其中一个指针遍历到数组末尾时，将另一个数组中剩余的元素添加到`merged`中。
- 根据`merged`的长度确定中位数的位置。如果`merged`的长度为奇数，则中位数为`merged`中间位置的元素；如果`merged`的长度为偶数，则中位数为中间两个位置的元素的平均值。
- 返回中位数。

----

## 5.存在重复元素

难度：<span style="background-color: #D62E04; color: white; font-family: Arial, sans-serif; font-size: 14px; font-weight: bold; padding: 3px 10px; border-radius: 8px;">困难</span>

> 给你一个整数数组 `nums` 和两个整数 `indexDiff` 和 `valueDiff`，找出满足下述条件的下标对 `(i, j)`：
>
> - `i != j`
> - `abs(i - j) <= indexDiff`
> - `abs(nums[i] - nums[j]) <= valueDiff`
>
> 如果存在满足条件的下标对，返回 `True`；否则，返回 `False`。

函数签名：

```python
from typing import List

def contains_nearby_almost_duplicate(nums: List[int], indexDiff: int, valueDiff: int) -> bool:
    pass
```

输入：

- nums：一个整数数组。
- indexDiff：一个整数，表示下标之差的最大值。
- valueDiff：一个整数，表示元素之差的最大值。

输出：

- 返回一个布尔值，表示是否存在满足条件的下标对。

要求：

- 如果输入参数不符合要求，函数应抛出异常。

示例代码：

```python
from typing import List

def contains_nearby_almost_duplicate(nums: List[int], indexDiff: int, valueDiff: int) -> bool:
    # 检查输入参数是否符合要求
    if not isinstance(nums, list) or not all(isinstance(num, int) for num in nums):
        raise ValueError("nums必须是由整数组成的列表")
    if not isinstance(indexDiff, int) or indexDiff < 0:
        raise ValueError("indexDiff必须是非负整数")
    if not isinstance(valueDiff, int) or valueDiff < 0:
        raise ValueError("valueDiff必须是非负整数")

    # 使用滑动窗口解决问题
    window = set()
    for i in range(len(nums)):
        if i > indexDiff:
            window.remove(nums[i - indexDiff - 1])
        for j in window:
            if abs(j - nums[i]) <= valueDiff:
                return True
        window.add(nums[i])
    
    return False


# 测试示例
print(contains_nearby_almost_duplicate([1, 2, 3, 1], 3, 0))
# 输出：True

print(contains_nearby_almost_duplicate([1, 0, 1, 1], 1, 2))
# 输出：True

print(contains_nearby_almost_duplicate([1, 5, 9, 1, 5, 9], 2, 3))
# 输出：False
```

说明：

- 在`contains_nearby_almost_duplicate`函数中，首先检查输入参数`nums`、`indexDiff`和`valueDiff`是否符合要求，如果不是则抛出异常。
- 使用滑动窗口的方法来解决问题。维护一个集合`window`，它表示当前窗口内的元素。遍历数组`nums`，对于每个元素，将其添加到`window`中，并判断是否存在满足条件的下标对。如果存在，则返回`True`；否则，将窗口最左侧的元素移除，并继续遍历下一个元素。
- 返回`False`，表示不存在满足条件的下标对。


