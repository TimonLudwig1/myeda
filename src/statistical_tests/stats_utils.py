import scipy.stats
import math
from math import sqrt

def significance_check(p_value: float, critical_level: float):
    if p_value < critical_level:
        stat_significance = f"statistically significant at the {critical_level * 100}% level"
        is_significant = True
    else:
        stat_significance = f"not statistically significant at the {critical_level * 100}% level"
        is_significant = False

    return stat_significance, is_significant

def calculate_confidence_interval(mean: float, t_crit: float, std: float | None = None, n: float | None = None, standard_error: float | None = None, type: str | None = "two") -> tuple[float, float]:

    if standard_error is None:
        if std is None or n is None: 
            raise ValueError("Missing required arguments")
        standard_error = (std / sqrt(n))

    margin = abs(t_crit) * standard_error

    if type == "two":
        return round(mean - margin, 4), round(mean + margin, 4)
    elif type == "right": 
        return round(mean - margin, 4), math.inf
    elif type == "left":
        return -math.inf, round(mean + margin, 4)
    raise ValueError(f"Unknown type: {type}")

def format_ci(lower: float, upper: float) -> str:
    left = "(" if math.isinf(lower) else "["
    right = ")" if math.isinf(upper) else "]"
    return f"{left}{lower} , {upper}{right}"
  

def _tailed_test(two_tailed: bool, left_tailed: bool, right_tailed: bool,t_statistic, dof, critical_level: float, mean: float, standard_error: float):
    if two_tailed: 
        t_crit = float(scipy.stats.t.ppf((1 - (critical_level/2)), dof))
        p_value = float(2 * scipy.stats.t.sf(abs(t_statistic), dof))
        stat_significance, is_significant =  significance_check(p_value, critical_level)
        test_type = "two_tailed"
        lower_bound, upper_bound = calculate_confidence_interval(mean=mean, t_crit=t_crit, standard_error=standard_error, type="two")   

    elif left_tailed: # mean1 significantly smaller than mean2 
        t_crit = float(scipy.stats.t.ppf((critical_level), dof))
        p_value = float(scipy.stats.t.cdf(t_statistic, dof))
        stat_significance, is_significant = significance_check(p_value, critical_level)
        test_type = "left_tailed"
        lower_bound, upper_bound = calculate_confidence_interval(mean=mean, t_crit=t_crit, standard_error=standard_error, type="left")   

    elif right_tailed: # mean1 significantly larger than mean2 
        t_crit = float(scipy.stats.t.ppf((1 - critical_level), dof))
        p_value = float(scipy.stats.t.sf(t_statistic, dof))
        stat_significance, is_significant = significance_check(p_value, critical_level)
        test_type = "right_tailed"
        lower_bound, upper_bound = calculate_confidence_interval(mean=mean, t_crit=t_crit, standard_error=standard_error, type="right")   

    else:
        raise ValueError("at least one test type has to be chosen. If you're using a dict, test type has to be passed as a seperate argument outside the dictionary")
    
    ci = format_ci(lower_bound, upper_bound)

    return t_crit, p_value, stat_significance, is_significant, test_type, lower_bound, upper_bound, ci

def f_test(k: float, N: float, within_group_variance: float, between_group_variance: float, critical_level: float = 0.05, two_tailed: bool = False, left_tailed: bool = False, right_tailed: bool = False):
    f_statistic = between_group_variance / within_group_variance
    dof_between = k - 1
    dof_within = N - k

    if two_tailed:
        f_crit = float(scipy.stats.f.ppf(1 - (critical_level/2), dof_between, dof_within))
        p_value = float(2* scipy.stats.f.sf(f_statistic, dof_between, dof_within))
        stat_significance, is_significant = significance_check(p_value, critical_level)
        test_type = "two_tailed"
    elif left_tailed:
        f_crit = float(scipy.stats.f.ppf((critical_level), dof_between, dof_within))
        p_value = float(scipy.stats.f.cdf(f_statistic, dof_between, dof_within))
        stat_significance, is_significant = significance_check(p_value, critical_level)
        test_type = "left_tailed"
    elif right_tailed:
        f_crit = float(scipy.stats.f.ppf((1 - critical_level), dof_between, dof_within))
        p_value = float(scipy.stats.f.sf(f_statistic, dof_between, dof_within))
        stat_significance, is_significant = significance_check(p_value, critical_level)
        test_type = "right_tailed"
    else:
        raise ValueError("at least one test type has to be chosen. If you're using a dict, test type has to be passed as a seperate argument outside the dictionary")

    return f_statistic, f_crit, p_value, stat_significance, is_significant, test_type 