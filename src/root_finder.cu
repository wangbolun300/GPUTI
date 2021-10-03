#include <gputi/root_finder.h>
#include <gputi/queue.h>
#include <iostream>

CCDdata array_to_ccd(std::array<std::array<Scalar, 3>, 8> a, bool is_edge)
{
    CCDdata data;
#pragma unroll
    for (int i = 0; i < 3; i++)
    {
        data.v0s[i] = a[0][i];
        data.v1s[i] = a[1][i];
        data.v2s[i] = a[2][i];
        data.v3s[i] = a[3][i];
        data.v0e[i] = a[4][i];
        data.v1e[i] = a[5][i];
        data.v2e[i] = a[6][i];
        data.v3e[i] = a[7][i];
    }
    return data;
}
__device__ __host__ VectorMax3d::VectorMax3d(Scalar a, Scalar b, Scalar c)
{
    v[0] = a;
    v[1] = b;
    v[2] = c;
}
__device__ __host__ void VectorMax3d::init(Scalar a, Scalar b, Scalar c){
    v[0] = a;
    v[1] = b;
    v[2] = c;
}
__device__ long reduction(const long n, long &result)
{
    int t = 0;
    int newn = n;
    while (newn % 2 == 0)
    {
        newn = newn / 2;
        t++;
    }
    result = newn;
    return t;
}

__device__ Singleinterval::Singleinterval(Numccd f, Numccd s)
{
    first = f;
    second = s;
}
__device__ interval_pair::interval_pair(const Singleinterval &a, const Singleinterval &b)
{
    first = a;
    second = b;
}

// calculate a*(2^b)
__device__ long power(const long a, const int b) { return a << b; }
__device__ Scalar Numccd2double(const Numccd &n)
{
    Scalar r = Scalar(n.first) / power(1, n.second);
    return r;
}
__device__ Numccd::Numccd(const long a, const int b)
{
    first = a;
    second = b;
}
__device__ bool sum_no_larger_1(const Numccd &num1, const Numccd &num2)
{
    long k1 = num1.first;
    int n1 = num1.second;
    long k2 = num2.first;
    int n2 = num2.second;
    long k;
    int n;
    if (n1 == n2)
    {
        k = k1 + k2;
        n = n1;
    }
    if (n1 < n2)
    {
        k = power(1, n2 - n1) * k1 + k2;
        n = n2;
    }
    if (n1 > n2)
    {
        k = power(1, n1 - n2) * k2 + k1;
        n = n1;
    }
    if (k > power(1, n))
        return false;
    else
        return true;
}
__device__ bool less_than(const Numccd &num1, const Numccd &num2)
{
    long k1 = num1.first;
    int n1 = num1.second;
    long k2 = num2.first;
    int n2 = num2.second;

    if (n1 < n2)
    {
        k1 = power(1, n2 - n1) * k1;
    }
    if (n1 > n2)
    {
        k2 = power(1, n1 - n2) * k2;
    }
    if (k1 < k2)
        return true;
    return false;
}
__device__ bool interval_overlap_region(
    const Singleinterval &itv, const Scalar r1, const Scalar r2)
{
    Scalar b1 = Numccd2double(itv.first);
    Scalar b2 = Numccd2double(itv.second);
    if (b2 < r1 || b1 > r2)
        return false;
    return true;
}

__device__ VectorMax3d width(const Singleinterval *x)
{
    VectorMax3d w;
#pragma unroll
    for (int i = 0; i < 3; i++)
    {
        w.v[i] =
            (Numccd2double(x[i].second) - Numccd2double(x[i].first));
    }
    return w;
}

__device__ void convert_tuv_to_array(
    const Singleinterval &itv0, const Singleinterval &itv1, Singleinterval &itv2,
    Scalar t_up[8],
    Scalar t_dw[8],
    Scalar u_up[8],
    Scalar u_dw[8],
    Scalar v_up[8],
    Scalar v_dw[8])
{
    // t order: 0,0,0,0,1,1,1,1
    // u order: 0,0,1,1,0,0,1,1
    // v order: 0,1,0,1,0,1,0,1
    Scalar t0_up = itv0.first.first,
           t0_dw = power(1, itv0.first.second),
           t1_up = itv0.second.first,
           t1_dw = power(1, itv0.second.second),

           u0_up = itv1.first.first,
           u0_dw = power(1, itv1.first.second),
           u1_up = itv1.second.first,
           u1_dw = power(1, itv1.second.second),

           v0_up = itv2.first.first,
           v0_dw = power(1, itv2.first.second),
           v1_up = itv2.second.first,
           v1_dw = power(1, itv2.second.second);

    t_up[0] = t0_up;
    t_up[1] = t0_up;
    t_up[2] = t0_up;
    t_up[3] = t0_up;
    t_up[4] = t1_up;
    t_up[5] = t1_up;
    t_up[6] = t1_up;
    t_up[7] = t1_up;
    t_dw[0] = t0_dw;
    t_dw[1] = t0_dw;
    t_dw[2] = t0_dw;
    t_dw[3] = t0_dw;
    t_dw[4] = t1_dw;
    t_dw[5] = t1_dw;
    t_dw[6] = t1_dw;
    t_dw[7] = t1_dw;
    u_up[0] = u0_up;
    u_up[1] = u0_up;
    u_up[2] = u1_up;
    u_up[3] = u1_up;
    u_up[4] = u0_up;
    u_up[5] = u0_up;
    u_up[6] = u1_up;
    u_up[7] = u1_up;
    u_dw[0] = u0_dw;
    u_dw[1] = u0_dw;
    u_dw[2] = u1_dw;
    u_dw[3] = u1_dw;
    u_dw[4] = u0_dw;
    u_dw[5] = u0_dw;
    u_dw[6] = u1_dw;
    u_dw[7] = u1_dw;
    v_up[0] = v0_up;
    v_up[1] = v1_up;
    v_up[2] = v0_up;
    v_up[3] = v1_up;
    v_up[4] = v0_up;
    v_up[5] = v1_up;
    v_up[6] = v0_up;
    v_up[7] = v1_up;
    v_dw[0] = v0_dw;
    v_dw[1] = v1_dw;
    v_dw[2] = v0_dw;
    v_dw[3] = v1_dw;
    v_dw[4] = v0_dw;
    v_dw[5] = v1_dw;
    v_dw[6] = v0_dw;
    v_dw[7] = v1_dw;
}

__device__ void function_vf(
    const Scalar &vs,
    const Scalar &t0s,
    const Scalar &t1s,
    const Scalar &t2s,
    const Scalar &ve,
    const Scalar &t0e,
    const Scalar &t1e,
    const Scalar &t2e,
    const Scalar t_up[8],
    const Scalar t_dw[8],
    const Scalar u_up[8],
    const Scalar u_dw[8],
    const Scalar v_up[8],
    const Scalar v_dw[8],
    Scalar rst[8])
{
    Scalar v;
    Scalar t0;
    Scalar t1;
    Scalar t2;
    Scalar pt;
#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        v = (ve - vs) * t_up[i] / t_dw[i] + vs;
        t0 = (t0e - t0s) * t_up[i] / t_dw[i] + t0s;
        t1 = (t1e - t1s) * t_up[i] / t_dw[i] + t1s;
        t2 = (t2e - t2s) * t_up[i] / t_dw[i] + t2s;
        pt = (t1 - t0) * u_up[i] / u_dw[i] + (t2 - t0) * v_up[i] / v_dw[i] + t0;
        rst[i] = v - pt;
    }
}

__device__ void function_ee(
    const Scalar &a0s,
    const Scalar &a1s,
    const Scalar &b0s,
    const Scalar &b1s,
    const Scalar &a0e,
    const Scalar &a1e,
    const Scalar &b0e,
    const Scalar &b1e,
    const Scalar t_up[8],
    const Scalar t_dw[8],
    const Scalar u_up[8],
    const Scalar u_dw[8],
    const Scalar v_up[8],
    const Scalar v_dw[8],
    Scalar rst[8])
{
#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        Scalar edge0_vertex0 = (a0e - a0s) * t_up[i] / t_dw[i] + a0s;
        Scalar edge0_vertex1 = (a1e - a1s) * t_up[i] / t_dw[i] + a1s;
        Scalar edge1_vertex0 = (b0e - b0s) * t_up[i] / t_dw[i] + b0s;
        Scalar edge1_vertex1 = (b1e - b1s) * t_up[i] / t_dw[i] + b1s;

        Scalar edge0_vertex =
            (edge0_vertex1 - edge0_vertex0) * u_up[i] / u_dw[i] + edge0_vertex0;
        Scalar edge1_vertex =
            (edge1_vertex1 - edge1_vertex0) * v_up[i] / v_dw[i] + edge1_vertex0;
        rst[i] = edge0_vertex - edge1_vertex;
    }
}

// ** this version can return the true x or y or z tolerance of the co-domain **
// eps is the interval [-eps,eps] we need to check
// if [-eps,eps] overlap, return true
// bbox_in_eps tell us if the box is totally in eps box
// ms is the minimum seperation
__device__ void evaluate_bbox_one_dimension_vector_return_tolerance(
    Scalar t_up[8],
    Scalar t_dw[8],
    Scalar u_up[8],
    Scalar u_dw[8],
    Scalar v_up[8],
    Scalar v_dw[8],
    const Scalar a0s[3],
    const Scalar a1s[3],
    const Scalar b0s[3],
    const Scalar b1s[3],
    const Scalar a0e[3],
    const Scalar a1e[3],
    const Scalar b0e[3],
    const Scalar b1e[3],
    int dimension,
    const bool &check_vf,
    const Scalar &eps,
    const Scalar &ms,
    bool &bbox_in_eps,
    Scalar &tol,
    bool &result)
{

    Scalar vs[8];
    bbox_in_eps = false;

#ifndef CHECK_EE
    function_vf(
        a0s[dimension], a1s[dimension], b0s[dimension], b1s[dimension],
        a0e[dimension], a1e[dimension], b0e[dimension], b1e[dimension],
        t_up, t_dw, u_up, u_dw, v_up, v_dw, vs);
#else
    function_ee(
        a0s[dimension], a1s[dimension], b0s[dimension], b1s[dimension],
        a0e[dimension], a1e[dimension], b0e[dimension], b1e[dimension],
        t_up, t_dw, u_up, u_dw, v_up, v_dw, vs);
#endif

    Scalar minv = vs[0];
    Scalar maxv = vs[0];
    minv = fminf(vs[1], minv);
    maxv = fmaxf(vs[1], maxv);
    minv = fminf(vs[2], minv);
    maxv = fmaxf(vs[2], maxv);
    minv = fminf(vs[3], minv);
    maxv = fmaxf(vs[3], maxv);
    minv = fminf(vs[4], minv);
    maxv = fmaxf(vs[4], maxv);
    minv = fminf(vs[5], minv);
    maxv = fmaxf(vs[5], maxv);
    minv = fminf(vs[6], minv);
    maxv = fmaxf(vs[6], maxv);
    minv = fminf(vs[7], minv);
    maxv = fmaxf(vs[7], maxv);

    tol = maxv - minv; // this is the real tolerance

    if (minv - ms > eps || maxv + ms < -eps)
    {
        result = false;
        return;
    }

    if (minv + ms >= -eps && maxv - ms <= eps)
    {
        bbox_in_eps = true;
    }
    result = true;
    return;
}

__device__ void bisect(const Singleinterval &inter, interval_pair &out)
{
    Numccd low = inter.first;
    Numccd up = inter.second;

    // interval is [k1/pow(2,n1), k2/pow(2,n2)], k1,k2,n1,n2 are all not negative
    long k1 = low.first;
    int n1 = low.second;
    long k2 = up.first;
    int n2 = up.second;

    long k;
    int n;
    int p;
    long r;
    if (n2 == n1)
    {
        p = reduction(k1 + k2, r);
        k = r;
        n = n2 - p + 1;
    }
    if (n2 > n1)
    {
        k = k1 * power(1, n2 - n1) + k2;
        n = n2 + 1;
    }
    if (n2 < n1)
    {
        k = k1 + k2 * power(1, n1 - n2);
        n = n1 + 1;
    }
    Numccd newnum(k, n);
    Singleinterval i1(low, newnum), i2(newnum, up);
    out.first = i1;
    out.second = i2;
}

__device__ bool interval_root_finder_double_horizontal_tree(
    const Scalar tol[3],
    const Scalar &co_domain_tolerance,
    const Singleinterval iset[3],
    const bool &check_t_overlap,
    const Scalar
        &max_t, // check interval [0, max_t] when check_t_overlap is set as TRUE
    Scalar &toi,
    const bool &check_vf,
    const Scalar err[3],
    const Scalar &ms,
    const Scalar a0s[3],
    const Scalar a1s[3],
    const Scalar b0s[3],
    const Scalar b1s[3],
    const Scalar a0e[3],
    const Scalar a1e[3],
    const Scalar b0e[3],
    const Scalar b1e[3],
    const int &max_itr,
    Scalar &output_tolerance,
    int &overflow_flag)
{

    overflow_flag = NO_OVERFLOW;
    // if max_itr <0, output_tolerance= co_domain_tolerance;
    // else, output_tolearance will be the precision after iteration time > max_itr
    output_tolerance = co_domain_tolerance;

    // this is used to catch the tolerance for each level
    Scalar temp_output_tolerance = co_domain_tolerance;

    // current intervals
    Singleinterval current[3];
    Scalar true_tol[3];

    // LINENBR 2
    int refine = 0;

    toi = SCALAR_LIMIT; //set toi as infinate
    // temp_toi is to catch the toi of each level
    Scalar temp_toi = toi;
    Numccd TOI;
    TOI.first = 4;
    TOI.second = 0; // set TOI as 4. this is to record the impact time of this level
    Numccd TOI_SKIP =
        TOI;               // this is to record the element that already small enough or contained in eps-box
    bool use_skip = false; // this is to record if TOI_SKIP is used.

    int current_level = -2; // in the begining, current_level != level
    int box_in_level = -2;  // this checks if all the boxes before this
    // level < tolerance. only true, we can return when we find one overlaps eps box and smaller than tolerance or eps-box
    bool this_level_less_tol = true;
    bool find_level_root = false;
    // Scalar current_tolerance=std::numeric_limits<Scalar>::infinity(); // set returned tolerance as infinite
    Scalar t_upper_bound = max_t; // 2*tol make it more conservative

    MinHeap istack;

    bool zero_in;
    bool box_in;
    Scalar t_up[8];
    Scalar t_dw[8];
    Scalar u_up[8];
    Scalar u_dw[8];
    Scalar v_up[8];
    Scalar v_dw[8];
    int level;
    bool box_in_[3];
    bool ck0, ck1, ck2;
    Singleinterval itv0, itv1, itv2;

    // LINENBR 3
    istack.insertKey(item(iset, -1));
    //return true;
    item current_item;

    //LINENBR 5
    while (!istack.empty())
    {
        if (overflow_flag != NO_OVERFLOW)
        {
            break;
        }

        //LINENBR 6
        current_item = istack.extractMin();

        current[0] = current_item.itv[0];
        current[1] = current_item.itv[1];
        current[2] = current_item.itv[2];
        level = current_item.level;

        // if this box is later than TOI_SKIP in time, we can skip this one.
        // TOI_SKIP is only updated when the box is small enough or totally contained in eps-box
        if (!less_than(current[0].first, TOI_SKIP))
        {
            continue;
        }
        if (box_in_level != level)
        { // before check a new level, set this_level_less_tol=true
            box_in_level = level;
            this_level_less_tol = true;
        }
        // LINENBR 8
        refine++;

        box_in = false;
        zero_in = false;

        itv0 = current[0];
        itv1 = current[1];
        itv2 = current[2];

        convert_tuv_to_array(itv0, itv1, itv2, t_up, t_dw, u_up, u_dw, v_up, v_dw);
        // LINENBR 7
        evaluate_bbox_one_dimension_vector_return_tolerance(
            t_up, t_dw, u_up, u_dw, v_up, v_dw, a0s, a1s, b0s, b1s, a0e,
            a1e, b0e, b1e, 0, check_vf, err[0], ms, box_in_[0],
            true_tol[0], ck0);
        evaluate_bbox_one_dimension_vector_return_tolerance(
            t_up, t_dw, u_up, u_dw, v_up, v_dw, a0s, a1s, b0s, b1s, a0e,
            a1e, b0e, b1e, 1, check_vf, err[1], ms, box_in_[1],
            true_tol[1], ck1);
        evaluate_bbox_one_dimension_vector_return_tolerance(
            t_up, t_dw, u_up, u_dw, v_up, v_dw, a0s, a1s, b0s, b1s, a0e,
            a1e, b0e, b1e, 2, check_vf, err[2], ms, box_in_[2],
            true_tol[2], ck2);

        box_in = box_in_[0] && box_in_[1] && box_in_[2];
        zero_in = ck0 && ck1 && ck2;

        if (!zero_in)
            continue;

        VectorMax3d widths = width(current);

        bool tol_condition = true_tol[0] <= co_domain_tolerance && true_tol[1] <= co_domain_tolerance && true_tol[2] <= co_domain_tolerance;

        // Condition 1, stopping condition on t, u and v is satisfied. this is useless now since we have condition 2
        bool condition1 = widths.v[0] <= tol[0] && widths.v[1] <= tol[1] && widths.v[2] <= tol[2];

        // Condition 2, zero_in = true, box inside eps-box and in this level,
        // no box whose zero_in is true but box size larger than tolerance, can return
        bool condition2 = box_in && this_level_less_tol;
        if (!tol_condition)
        {
            this_level_less_tol = false;
            // this level has at least one box whose size > tolerance, thus we
            // cannot directly return if find one box whose size < tolerance or box-in
        }

        // Condition 3, in this level, we find a box that zero-in and size < tolerance.
        // and no other boxes whose zero-in is true in this level before this one is larger than tolerance, can return
        bool condition3 = this_level_less_tol;
        // LINENBR 15, 16
        if (condition1 || condition2 || condition3)
        {
            TOI = current[0].first;

            // continue;
            toi = Numccd2double(TOI);

            return true;
        }

        // LINENBR 10
        if (current_level != level)
        {
            // LINENBR 22
            current_level = level;
            find_level_root = false;
        }
        if (!find_level_root)
        {
            TOI = current[0].first;

            // LINENBR 11
            // continue;
            temp_toi = Numccd2double(TOI);

            // if the real tolerance is larger than input, use the real one;
            // if the real tolerance is smaller than input, use input
            temp_output_tolerance = max(
                max(
                    max(true_tol[0], true_tol[1]), true_tol[2]),
                co_domain_tolerance);

            find_level_root =
                true; // this ensures always find the earlist root
        }

        // LINENBR 12
        if (refine > max_itr)
        {
            overflow_flag = ITERATION_OVERFLOW;
            break;
        }

        // if this box is small enough, or inside of eps-box, then just continue,
        // but we need to record the collision time
        if (tol_condition || box_in)
        {
            if (less_than(current[0].first, TOI_SKIP))
            {
                TOI_SKIP = current[0].first;
            }
            use_skip = true;
            continue;
        }

        bool check[3];
        VectorMax3d widthratio;

        check[0] = false;
        check[1] = false;
        check[2] = false;
        for (int i = 0; i < 3; i++)
        {
            widthratio.v[i] = widths.v[i] / tol[i];
            if (widths.v[i] > tol[i])
                check[i] = true; // means this need to be checked
        }

        int split_i = -1;

        for (int i = 0; i < 3; i++)
        {
            if (check[i])
            {
                if (check[(i + 1) % 3] && check[(i + 2) % 3])
                {
                    if (widthratio.v[i] >= widthratio.v[(i + 1) % 3] && widthratio.v[i] >= widthratio.v[(i + 2) % 3])
                    {
                        split_i = i;
                        break;
                    }
                }
                if (check[(i + 1) % 3] && !check[(i + 2) % 3])
                {
                    if (widthratio.v[i] >= widthratio.v[(i + 1) % 3])
                    {
                        split_i = i;
                        break;
                    }
                }
                if (!check[(i + 1) % 3] && check[(i + 2) % 3])
                {
                    if (widthratio.v[i] >= widthratio.v[(i + 2) % 3])
                    {
                        split_i = i;
                        break;
                    }
                }
                if (!check[(i + 1) % 3] && !check[(i + 2) % 3])
                {

                    split_i = i;
                    break;
                }
            }
        }

        interval_pair halves;
        Singleinterval bisect_inter = current[split_i];
        // LINENBR 19
        bisect(bisect_inter, halves);
        if (!less_than(halves.first.first, halves.first.second))
        {
            overflow_flag = BISECTION_OVERFLOW;
            break;
        }
        if (!less_than(halves.second.first, halves.second.second))
        {
            overflow_flag = BISECTION_OVERFLOW;
            break;
        }
#ifndef CHECK_EE // check vf

        if (split_i == 1)
        {

            if (sum_no_larger_1(halves.second.first, current[2].first))
            {

                current[split_i] = halves.second;
                // LINENBR 20
                bool inserted = istack.insertKey(item(current, level + 1));
                if (inserted == false)
                {
                    overflow_flag = HEAP_OVERFLOW;
                }
            }
            if (sum_no_larger_1(halves.first.first, current[2].first))
            {
                current[split_i] = halves.first;
                bool inserted = istack.insertKey(item(current, level + 1));
                if (inserted == false)
                {
                    overflow_flag = HEAP_OVERFLOW;
                }
            }
        }

        if (split_i == 2)
        {

            if (sum_no_larger_1(halves.second.first, current[1].first))
            {
                current[split_i] = halves.second;
                bool inserted = istack.insertKey(item(current, level + 1));
                if (inserted == false)
                {
                    overflow_flag = HEAP_OVERFLOW;
                }
            }
            if (sum_no_larger_1(halves.first.first, current[1].first))
            {
                current[split_i] = halves.first;
                bool inserted = istack.insertKey(item(current, level + 1));
                if (inserted == false)
                {
                    overflow_flag = HEAP_OVERFLOW;
                }
            }
        }
        if (split_i == 0)
        {
            if (check_t_overlap)
            {
                if (interval_overlap_region(
                        halves.second, 0, t_upper_bound))
                {
                    current[split_i] = halves.second;
                    bool inserted = istack.insertKey(item(current, level + 1));
                    if (inserted == false)
                    {
                        overflow_flag = HEAP_OVERFLOW;
                    }
                }
                if (interval_overlap_region(
                        halves.first, 0, t_upper_bound))
                {
                    current[split_i] = halves.first;
                    bool inserted = istack.insertKey(item(current, level + 1));
                    if (inserted == false)
                    {
                        overflow_flag = HEAP_OVERFLOW;
                    }
                }
            }
            else
            {
                current[split_i] = halves.second;
                bool inserted = istack.insertKey(item(current, level + 1));
                if (inserted == false)
                {
                    overflow_flag = HEAP_OVERFLOW;
                }
                current[split_i] = halves.first;
                inserted = istack.insertKey(item(current, level + 1));
                if (inserted == false)
                {
                    overflow_flag = HEAP_OVERFLOW;
                }
            }
        }

#else
        if (check_t_overlap && split_i == 0)
        {
            if (interval_overlap_region(
                    halves.second, 0, t_upper_bound))
            {
                current[split_i] = halves.second;
                bool inserted = istack.insertKey(item(current, level + 1));
                if (inserted == false)
                {
                    overflow_flag = HEAP_OVERFLOW;
                }
            }
            if (interval_overlap_region(halves.first, 0, t_upper_bound))
            {
                current[split_i] = halves.first;
                bool inserted = istack.insertKey(item(current, level + 1));
                if (inserted == false)
                {
                    overflow_flag = HEAP_OVERFLOW;
                }
            }
        }
        else
        {
            current[split_i] = halves.second;
            bool inserted = istack.insertKey(item(current, level + 1));
            if (inserted == false)
            {
                overflow_flag = HEAP_OVERFLOW;
            }
            current[split_i] = halves.first;
            inserted = istack.insertKey(item(current, level + 1));
            if (inserted == false)
            {
                overflow_flag = HEAP_OVERFLOW;
            }
        }
#endif
    }

    if (overflow_flag > 0)
    {
        toi = temp_toi;
        output_tolerance = temp_output_tolerance;
        return true;
    }

    if (use_skip)
    {
        toi = Numccd2double(TOI_SKIP);

        return true;
    }

    return false;
}

__device__ bool interval_root_finder_double_horizontal_tree(
    const Scalar tol[3],
    const Scalar &co_domain_tolerance,
    Scalar &toi,
    const bool check_vf,
    const Scalar err[3], // this is the maximum error on each axis when calculating the vertices, err, aka, filter
    const Scalar ms,
    const Scalar a0s[3],
    const Scalar a1s[3],
    const Scalar b0s[3],
    const Scalar b1s[3],
    const Scalar a0e[3],
    const Scalar a1e[3],
    const Scalar b0e[3],
    const Scalar b1e[3],
    const Scalar max_time,
    const int max_itr,
    Scalar &output_tolerance,
    int &overflow_flag)
{
    return true;
}

__device__ Scalar max_linf_dist(const VectorMax3d &p1, const VectorMax3d &p2)
{
    Scalar r = 0;
    r = max(r, fabs(p1.v[0] - p2.v[0]));
    r = max(r, fabs(p1.v[1] - p2.v[1]));
    r = max(r, fabs(p1.v[2] - p2.v[2]));
    return r;
}

__device__ Scalar max_linf_4(
    const VectorMax3d &p1,
    const VectorMax3d &p2,
    const VectorMax3d &p3,
    const VectorMax3d &p4,
    const VectorMax3d &p1e,
    const VectorMax3d &p2e,
    const VectorMax3d &p3e,
    const VectorMax3d &p4e)
{
    Scalar r = 0; //,temp = 0, temp2=0, temp3 = 0, temp4=0;
    r = max(r, max_linf_dist(p4e, p4));
    r = max(r, max_linf_dist(p3e, p3));
    r = max(r, max_linf_dist(p2e, p2));
    return max(r, max_linf_dist(p1e, p1));
}
__device__ void compute_face_vertex_tolerance_3d_new(
    const CCDdata &data_in,
    const Scalar tolerance, Scalar *result)
{
    VectorMax3d vs(data_in.v0s[0], data_in.v0s[1], data_in.v0s[2]);
    VectorMax3d f0s(data_in.v1s[0], data_in.v1s[1], data_in.v1s[2]);
    VectorMax3d f1s(data_in.v2s[0], data_in.v2s[1], data_in.v2s[2]);
    VectorMax3d f2s(data_in.v3s[0], data_in.v3s[1], data_in.v3s[2]);
    VectorMax3d ve(data_in.v0e[0], data_in.v0e[1], data_in.v0e[2]);
    VectorMax3d f0e(data_in.v1e[0], data_in.v1e[1], data_in.v1e[2]);
    VectorMax3d f1e(data_in.v2e[0], data_in.v2e[1], data_in.v2e[2]);
    VectorMax3d f2e(data_in.v3e[0], data_in.v3e[1], data_in.v3e[2]);
    VectorMax3d p000 = vs - f0s, p001 = vs - f2s,
                p011 = vs - (f1s + f2s - f0s), p010 = vs - f1s;
    VectorMax3d p100 = ve - f0e, p101 = ve - f2e,
                p111 = ve - (f1e + f2e - f0e), p110 = ve - f1e;
    Scalar dl = 0;
    Scalar edge0_length = 0;
    Scalar edge1_length = 0;
    dl = 3 * max_linf_4(p000, p001, p011, p010, p100, p101, p111, p110);
    edge0_length =
        3 * max_linf_4(p000, p100, p101, p001, p010, p110, p111, p011);
    edge1_length =
        3 * max_linf_4(p000, p100, p110, p010, p001, p101, p111, p011);

    result[0] = tolerance / dl;
    result[1] = tolerance / edge0_length;
    result[2] = tolerance / edge1_length;
}
__device__ void compute_edge_edge_tolerance_new(
    const CCDdata &data_in,
    const Scalar tolerance, Scalar *result)
{
    VectorMax3d edge0_vertex0_start(data_in.v0s[0], data_in.v0s[1], data_in.v0s[2]); // a0s
    VectorMax3d edge0_vertex1_start(data_in.v1s[0], data_in.v1s[1], data_in.v1s[2]); // a1s
    VectorMax3d edge1_vertex0_start(data_in.v2s[0], data_in.v2s[1], data_in.v2s[2]); // b0s
    VectorMax3d edge1_vertex1_start(data_in.v3s[0], data_in.v3s[1], data_in.v3s[2]); // b1s
    VectorMax3d edge0_vertex0_end(data_in.v0e[0], data_in.v0e[1], data_in.v0e[2]);
    VectorMax3d edge0_vertex1_end(data_in.v1e[0], data_in.v1e[1], data_in.v1e[2]);
    VectorMax3d edge1_vertex0_end(data_in.v2e[0], data_in.v2e[1], data_in.v2e[2]);
    VectorMax3d edge1_vertex1_end(data_in.v3e[0], data_in.v3e[1], data_in.v3e[2]);
    VectorMax3d p000 = edge0_vertex0_start - edge1_vertex0_start,
                p001 = edge0_vertex0_start - edge1_vertex1_start,
                p011 = edge0_vertex1_start - edge1_vertex1_start,
                p010 = edge0_vertex1_start - edge1_vertex0_start;
    VectorMax3d p100 = edge0_vertex0_end - edge1_vertex0_end,
                p101 = edge0_vertex0_end - edge1_vertex1_end,
                p111 = edge0_vertex1_end - edge1_vertex1_end,
                p110 = edge0_vertex1_end - edge1_vertex0_end;
    Scalar dl = 0;
    Scalar edge0_length = 0;
    Scalar edge1_length = 0;

    dl = 3 * max_linf_4(p000, p001, p011, p010, p100, p101, p111, p110);
    edge0_length =
        3 * max_linf_4(p000, p100, p101, p001, p010, p110, p111, p011);
    edge1_length =
        3 * max_linf_4(p000, p100, p110, p010, p001, p101, p111, p011);
    result[0] = tolerance / dl;
    result[1] = tolerance / edge0_length;
    result[2] = tolerance / edge1_length;
}

__device__ __host__ void get_numerical_error(
    const VectorMax3d vertices[8], const int vsize,
    const bool &check_vf,
    const bool using_minimum_separation,
    Scalar *error)
{
    Scalar eefilter;
    Scalar vffilter;
#ifdef NO_CHECK_MS

#ifdef GPUTI_USE_DOUBLE_PRECISION
    eefilter = 6.217248937900877e-15;
    vffilter = 6.661338147750939e-15;
#else
    eefilter = 3.337861e-06;
    vffilter = 3.576279e-06;
#endif

#else
#ifdef GPUTI_USE_DOUBLE_PRECISION
    eefilter = 7.105427357601002e-15;
    vffilter = 7.549516567451064e-15;
#else
    eefilter = 3.814698e-06;
    vffilter = 4.053116e-06;
#endif

#endif

    Scalar xmax = fabs(vertices[0].v[0]);
    Scalar ymax = fabs(vertices[0].v[1]);
    Scalar zmax = fabs(vertices[0].v[2]);

    for (int i = 0; i < vsize; i++)
    {
        if (xmax < fabs(vertices[i].v[0]))
        {
            xmax = fabs(vertices[i].v[0]);
        }
        if (ymax < fabs(vertices[i].v[1]))
        {
            ymax = fabs(vertices[i].v[1]);
        }
        if (zmax < fabs(vertices[i].v[2]))
        {
            zmax = fabs(vertices[i].v[2]);
        }
    }
    Scalar delta_x = xmax > 1 ? xmax : 1;
    Scalar delta_y = ymax > 1 ? ymax : 1;
    Scalar delta_z = zmax > 1 ? zmax : 1;
#ifdef CHECK_EE
    error[0] = delta_x * delta_x * delta_x * eefilter;
    error[1] = delta_y * delta_y * delta_y * eefilter;
    error[2] = delta_z * delta_z * delta_z * eefilter;
#else
    error[0] = delta_x * delta_x * delta_x * vffilter;
    error[1] = delta_y * delta_y * delta_y * vffilter;
    error[2] = delta_z * delta_z * delta_z * vffilter;
#endif
    return;
}

__device__ bool CCD_Solver(
    const CCDdata &data_in,
    const Scalar err[3],
    const Scalar ms,
    Scalar &toi,
    Scalar tolerance,
    Scalar t_max,
    const int max_itr,
    Scalar &output_tolerance,
    bool no_zero_toi,
    int &overflow_flag,
    bool is_vf)
{

    return true;
}
__device__ bool Origin_in_function_bounding_box_double_vector_return_tolerance(
    Singleinterval *paras,
    const Scalar *a0s,
    const Scalar *a1s,
    const Scalar *b0s,
    const Scalar *b1s,
    const Scalar *a0e,
    const Scalar *a1e,
    const Scalar *b0e,
    const Scalar *b1e,
    const bool check_vf,
    const Scalar *box,
    const Scalar ms,
    bool &box_in_eps,
    Scalar *tolerance)
{

    // box_in_eps = false;
    // Scalar t_up[8];
    // Scalar t_dw[8];
    // Scalar u_up[8];
    // Scalar u_dw[8];
    // Scalar v_up[8];
    // Scalar v_dw[8];
    // Singleinterval *itv0 = &paras[0], *itv1 = &paras[1], *itv2 = &paras[2];

    // convert_tuv_to_array(itv0, itv1, itv2, t_up, t_dw, u_up, u_dw, v_up, v_dw);

    // bool ck;
    // bool box_in[3];
    // for (int i = 0; i < 3; i++)
    // {

    //     ck = evaluate_bbox_one_dimension_vector_return_tolerance(
    //         t_up, t_dw, u_up, u_dw, v_up, v_dw, a0s, a1s, b0s, b1s, a0e,
    //         a1e, b0e, b1e, i, check_vf, box[i], ms, box_in[i],
    //         tolerance[i]);

    //     if (!ck)
    //         return false;
    // }
    // if (box_in[0] && box_in[1] && box_in[2])
    // {
    //     box_in_eps = true;
    // }
    return true;
}
__device__ void compute_face_vertex_tolerance(const CCDdata &data_in,const CCDConfig& config, CCDOut& out){
    VectorMax3d v(data_in.v0s[0], data_in.v0s[1], data_in.v0s[2]);
    VectorMax3d f0(data_in.v1s[0], data_in.v1s[1], data_in.v1s[2]);
    VectorMax3d f1(data_in.v2s[0], data_in.v2s[1], data_in.v2s[2]);
    VectorMax3d f2(data_in.v3s[0], data_in.v3s[1], data_in.v3s[2]);
    VectorMax3d p000 = v - f0, p001 = v - f2,
                p011 = v - (f1 + f2 - f0), p010 = v - f1;
    v.init(data_in.v0e[0], data_in.v0e[1], data_in.v0e[2]);
    f0.init(data_in.v1e[0], data_in.v1e[1], data_in.v1e[2]);
    f1.init(data_in.v2e[0], data_in.v2e[1], data_in.v2e[2]);
    f2.init(data_in.v3e[0], data_in.v3e[1], data_in.v3e[2]);   
    VectorMax3d p100 = v - f0, p101 = v - f2,
                p111 = v - (f1 + f2 - f0), p110 = v - f1;
    
    Scalar dl = 3 * max_linf_4(p000, p001, p011, p010, p100, p101, p111, p110);
    Scalar edge0_length =
        3 * max_linf_4(p000, p100, p101, p001, p010, p110, p111, p011);
    Scalar edge1_length =
        3 * max_linf_4(p000, p100, p110, p010, p001, p101, p111, p011);

    out.tol[0] = config.co_domain_tolerance / dl;
    out.tol[1] = config.co_domain_tolerance / edge0_length;
    out.tol[2] = config.co_domain_tolerance / edge1_length;
}

__device__ __host__ void get_numerical_error_vf(
    const CCDdata &data_in,
    BoxCompute &box)
{
    Scalar vffilter;

#ifdef GPUTI_USE_DOUBLE_PRECISION
    vffilter = 6.661338147750939e-15;
#else
    vffilter = 3.576279e-06;
#endif
    Scalar xmax = fabs(data_in.v0s[0]);
    Scalar ymax = fabs(data_in.v0s[1]);
    Scalar zmax = fabs(data_in.v0s[2]);

    xmax = max(xmax,fabs(data_in.v1s[0]));
    ymax = max(ymax,fabs(data_in.v1s[1]));
    zmax = max(zmax,fabs(data_in.v1s[2]));
    
    xmax = max(xmax,fabs(data_in.v2s[0]));
    ymax = max(ymax,fabs(data_in.v2s[1]));
    zmax = max(zmax,fabs(data_in.v2s[2]));

    xmax = max(xmax,fabs(data_in.v3s[0]));
    ymax = max(ymax,fabs(data_in.v3s[1]));
    zmax = max(zmax,fabs(data_in.v3s[2]));

    xmax = max(xmax,fabs(data_in.v0e[0]));
    ymax = max(ymax,fabs(data_in.v0e[1]));
    zmax = max(zmax,fabs(data_in.v0e[2]));

    xmax = max(xmax,fabs(data_in.v1e[0]));
    ymax = max(ymax,fabs(data_in.v1e[1]));
    zmax = max(zmax,fabs(data_in.v1e[2]));

    xmax = max(xmax,fabs(data_in.v2e[0]));
    ymax = max(ymax,fabs(data_in.v2e[1]));
    zmax = max(zmax,fabs(data_in.v2e[2]));

    xmax = max(xmax,fabs(data_in.v3e[0]));
    ymax = max(ymax,fabs(data_in.v3e[1]));
    zmax = max(zmax,fabs(data_in.v3e[2]));

    xmax = max(xmax, Scalar(1));
    ymax = max(ymax, Scalar(1));
    zmax = max(zmax, Scalar(1));

    box.err[0] = xmax * xmax * xmax * vffilter;
    box.err[1] = ymax * ymax * ymax * vffilter;
    box.err[2] = zmax * zmax * zmax * vffilter;
    return;
}
// Singleinterval *paras,
//     const Scalar *a0s,
//     const Scalar *a1s,
//     const Scalar *b0s,
//     const Scalar *b1s,
//     const Scalar *a0e,
//     const Scalar *a1e,
//     const Scalar *b0e,
//     const Scalar *b1e,
//     const bool check_vf,
//     const Scalar *box,
//     const Scalar ms,
//     bool &box_in_eps,
//     Scalar *tolerance)
__device__ void BoxPrimatives::calculate_tuv(const BoxCompute& box){
    if(b[0]==0){// t0
        t_up=box.current_item.itv[0].first.first;
        t_dw=1<<box.current_item.itv[0].first.second;
    }
    else{// t1
        t_up=box.current_item.itv[0].second.first;
        t_dw=1<<box.current_item.itv[0].second.second;
    }

    if(b[1]==0){// u0
        u_up=box.current_item.itv[1].first.first;
        u_dw=1<<box.current_item.itv[1].first.second;
    }
    else{// u1
        u_up=box.current_item.itv[1].second.first;
        u_dw=1<<box.current_item.itv[1].second.second;
    }

    if(b[2]==0){// v0
        v_up=box.current_item.itv[2].first.first;
        v_dw=1<<box.current_item.itv[2].first.second;
    }
    else{// v1
        v_up=box.current_item.itv[2].second.first;
        v_dw=1<<box.current_item.itv[2].second.second;
    }
}
__device__ Scalar calculate_vf(const CCDdata &data_in, const BoxPrimatives& bp){
    Scalar v, pt, t0, t1, t2;
    v = (data_in.v0e[bp.dim] - data_in.v0s[bp.dim]) * bp.t_up / bp.t_dw + data_in.v0s[bp.dim];
        t0 = (data_in.v1e[bp.dim] - data_in.v1s[bp.dim]) * bp.t_up / bp.t_dw + data_in.v1s[bp.dim];
        t1 = (data_in.v2e[bp.dim] - data_in.v2s[bp.dim]) * bp.t_up / bp.t_dw + data_in.v2s[bp.dim];
        t2 = (data_in.v3e[bp.dim] - data_in.v3s[bp.dim]) * bp.t_up / bp.t_dw + data_in.v3s[bp.dim];
        pt = (t1 - t0) * bp.u_up / bp.u_dw + (t2 - t0) * bp.v_up / bp.v_dw + t0;
        return (v - pt);
}

__device__ bool Origin_in_vf_inclusion_function(const CCDdata &data_in, BoxCompute& box){
    BoxPrimatives bp;
    Scalar vmin=-SCALAR_LIMIT;
    Scalar vmax=SCALAR_LIMIT;
    Scalar value;
    for(bp.dim=0;bp.dim<3;bp.dim++){

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                for (int k = 0; k < 2; k++)
                {
                    bp.b[0] = i;
                    bp.b[1] = j;
                    bp.b[2] = k; //100
                    bp.calculate_tuv(box);
                    value = calculate_vf(data_in, bp);
                    vmin = min(vmin, value);
                    vmax = max(vmax, value);
                }
            }
        }
        // get the min and max in one dimension
        box.true_tol = max(box.true_tol, vmax - vmin); // this is the real tolerance

        if (vmin > box.err[bp.dim] || vmax < -box.err[bp.dim])
        {
            return false;
        }

        if (vmin < -box.err[bp.dim] || vmax > box.err[bp.dim])
        {
            box.box_in = false;
        }
        
    }
    return true;
}
__device__ void calculate_interval_widths(BoxCompute& box)
{
    
    box.widths[0] = Scalar(box.current_item.itv[0].second.first) / (1 << box.current_item.itv[0].second.second) - Scalar(box.current_item.itv[0].first.first) / (1 << box.current_item.itv[0].first.second);
    box.widths[1] = Scalar(box.current_item.itv[1].second.first) / (1 << box.current_item.itv[1].second.second) - Scalar(box.current_item.itv[1].first.first) / (1 << box.current_item.itv[1].first.second);
    box.widths[2] = Scalar(box.current_item.itv[2].second.first) / (1 << box.current_item.itv[2].second.second) - Scalar(box.current_item.itv[2].first.first) / (1 << box.current_item.itv[2].first.second);
}
__device__ void split_dimension(const CCDOut& out,BoxCompute& box){
    Scalar res[3];
    res[0]=box.widths[0]/out.tol[0];
    res[1]=box.widths[1]/out.tol[1];
    res[2]=box.widths[2]/out.tol[2];
    if(res[0]>=res[1]&&res[0]>=res[2]){
        box.split=0;
    }
    if(res[1]>=res[0]&&res[1]>=res[2]){
        box.split=1;
    }
    if(res[2]>=res[1]&&res[2]>=res[0]){
        box.split=2;
    }
}

__device__ void bisect_vf_and_push(BoxCompute& box,const CCDConfig& config, MinHeap& istack,CCDOut& out){
    interval_pair halves;
    bool inserted;
    bisect(box.current_item.itv[box.split], halves);
    if (!less_than(halves.first.first, halves.first.second))
    {
        out.overflow_flag = BISECTION_OVERFLOW;
        return;
    }
    if (!less_than(halves.second.first, halves.second.second))
    {
        out.overflow_flag = BISECTION_OVERFLOW;
        return;
    }
    if (box.split == 0)// split t interval
    {
        if (config.max_t!=1)
        {
            if (interval_overlap_region(
                    halves.second, 0, config.max_t))
            {
                box.current_item.itv[box.split] = halves.second;
                inserted = istack.insertKey(item(box.current_item.itv, box.current_item.level + 1));
                if (inserted == false)
                {
                    out.overflow_flag = HEAP_OVERFLOW;
                }
            }
            if (interval_overlap_region(
                    halves.first, 0, config.max_t))
            {
                box.current_item.itv[box.split] = halves.first;
                inserted = istack.insertKey(item(box.current_item.itv, box.current_item.level + 1));
                if (inserted == false)
                {
                    out.overflow_flag = HEAP_OVERFLOW;
                }
            }
        }
        else
        {
            box.current_item.itv[box.split] = halves.second;
            inserted = istack.insertKey(item(box.current_item.itv, box.current_item.level + 1));
            if (inserted == false)
            {
                out.overflow_flag = HEAP_OVERFLOW;
            }
            box.current_item.itv[box.split] = halves.first;
            inserted = istack.insertKey(item(box.current_item.itv, box.current_item.level + 1));
            if (inserted == false)
            {
                out.overflow_flag = HEAP_OVERFLOW;
            }
        }
    }

    if (box.split == 1) // split u interval
    {

        if (sum_no_larger_1(halves.second.first, box.current_item.itv[2].first)) // check if u+v<=1
        {

            box.current_item.itv[box.split] = halves.second;
            // LINENBR 20
            inserted = istack.insertKey(item(box.current_item.itv, box.current_item.level + 1));
            if (inserted == false)
            {
                out.overflow_flag = HEAP_OVERFLOW;
            }
        }

        box.current_item.itv[box.split] = halves.first;
        inserted = istack.insertKey(item(box.current_item.itv, box.current_item.level + 1));
        if (inserted == false)
        {
            out.overflow_flag = HEAP_OVERFLOW;
        }
    }
    if (box.split == 2) // split v interval
    {
        if (sum_no_larger_1(halves.second.first, box.current_item.itv[1].first))
        {
            box.current_item.itv[box.split] = halves.second;
            inserted = istack.insertKey(item(box.current_item.itv, box.current_item.level + 1));
            if (inserted == false)
            {
                out.overflow_flag = HEAP_OVERFLOW;
            }
        }

        box.current_item.itv[box.split] = halves.first;
        inserted = istack.insertKey(item(box.current_item.itv, box.current_item.level + 1));
        if (inserted == false)
        {
            out.overflow_flag = HEAP_OVERFLOW;
        }
    }
}

__device__ bool vertexFaceCCD(const CCDdata &data_in,const CCDConfig& config, CCDOut& out){
    
    MinHeap istack;// now when initialized, size is 1 and initialized with [0,1]^3
    compute_face_vertex_tolerance(data_in, config, out);
    BoxCompute box;

#ifdef CALCULATE_ERROR_BOUND
    get_numerical_error_vf(data_in, box);
#else
    box.err[0] = config.err_in[0];
    box.err[1] = config.err_in[1];
    box.err[2] = config.err_in[2];
#endif

    out.output_tolerance = config.co_domain_tolerance;

    // this is used to catch the tolerance for each level
    Scalar temp_output_tolerance = config.co_domain_tolerance;
    // LINENBR 2
    int refine = 0;
    // temp_toi is to catch the first toi of each level
    Scalar temp_toi = out.toi;
    Scalar skip_toi =out.toi;
    
    bool use_skip = false; // when tolerance is small enough or when box in epsilon, this is activated.
    int current_level = -2; // in the begining, current_level != level
    int box_in_level = -2;  // this checks if all the boxes before this
    // level < tolerance. only true, we can return when we find one overlaps eps box and smaller than tolerance or eps-box
    bool this_level_less_tol = true;
    bool find_level_root = false;

    while (!istack.empty())
    {
        if (out.overflow_flag != NO_OVERFLOW)
        {
            break;
        }

        //LINENBR 6
        box.current_item = istack.extractMin();// get the level and the intervals
        box.current_toi=Scalar(box.current_item.itv[0].first.first) / (1 << box.current_item.itv[0].first.second);
        
        // if this box is later than TOI_SKIP in time, we can skip this one.
        // TOI_SKIP is only updated when the box is small enough or totally contained in eps-box
        if (box.current_toi>=skip_toi)
        {
            continue;
        }
        if (box_in_level != box.current_item.level)
        { // before check a new level, set this_level_less_tol=true
            box_in_level = box.current_item.level;
            this_level_less_tol = true;
        }
        // LINENBR 8
        refine++;
        bool zero_in =
            Origin_in_vf_inclusion_function(data_in,box);
        //return zero_in;// REGSCOUNT 100
        
        if (!zero_in)
            continue;

        
        calculate_interval_widths(box);
                
        // LINENBR 15, 16
        // Condition 1, stopping condition on t, u and v is satisfied. this is useless now since we have condition 2
        bool condition = box.widths[0] <= out.tol[0] && box.widths[1] <= out.tol[1] && box.widths[2] <= out.tol[2];
        if(condition){
            out.toi=box.current_toi;
            return true;
        }
        // Condition 2, zero_in = true, box inside eps-box and in this level,
        // no box whose zero_in is true but box size larger than tolerance, can return
        condition = box.box_in && this_level_less_tol;
        if(condition){
            out.toi=box.current_toi;
            return true;
        }

        bool tol_condition = box.true_tol <= config.co_domain_tolerance;
        if (!tol_condition)
        {
            this_level_less_tol = false;
            // this level has at least one box whose size > tolerance, thus we
            // cannot directly return if find one box whose size < tolerance or box-in
        }

        // Condition 3, in this level, we find a box that zero-in and size < tolerance.
        // and no other boxes whose zero-in is true in this level before this one is larger than tolerance, can return
        condition = this_level_less_tol;
        if(condition){
            out.toi=box.current_toi;
            return true;
        }

        // This is for early termination, finding the earlist root of this level in case of early termination happens
        if (current_level != box.current_item.level)
        {
            // LINENBR 22
            current_level = box.current_item.level;
            find_level_root = false;
        }
        if (!find_level_root)
        {
            // LINENBR 11
            // this is the first toi of this level
            temp_toi = box.current_toi;
            // if the real tolerance is larger than input, use the real one;
            // if the real tolerance is smaller than input, use input
            temp_output_tolerance = max(box.true_tol,config.co_domain_tolerance);
            find_level_root =true; // this ensures always find the earlist root
        }

        // LINENBR 12
        if (refine > config.max_itr)
        {
            out.overflow_flag = ITERATION_OVERFLOW;
            break;
        }

        // if this box is small enough, or inside of eps-box, then just continue,
        // but we need to record the collision time
        if (tol_condition || box.box_in )
        {
            if(box.current_toi<skip_toi)
            {
                skip_toi=box.current_toi;
            }
            use_skip = true;
            continue;
        }
        split_dimension(out,box);
        bisect_vf_and_push(box,config, istack,out);
    }
    if (out.overflow_flag != NO_OVERFLOW)
    {
        out.toi = temp_toi;
        out.output_tolerance = temp_output_tolerance;
        return true;
    }

    if (use_skip)
    {
        out.toi = skip_toi;

        return true;
    }
    return false;
}


__device__ bool vertexFaceCCD_double(
    const CCDdata &data_in,
    const Scalar *err_in,
    const Scalar ms,
    Scalar &toi,
    Scalar co_domain_tolerance,
    Scalar max_t,
    const int max_itr,
    Scalar &output_tolerance,
    bool no_zero_toi,
    int &overflow_flag)
{
    Scalar temp_output_tolerance;
    Scalar temp_toi;
    Scalar t_upper_bound;
    Scalar tol[3];
    Scalar err[3];
    Scalar true_tol[3];
    Scalar t_up[8];
    Scalar t_dw[8];
    Scalar u_up[8];
    Scalar u_dw[8];
    Scalar v_up[8];
    Scalar v_dw[8];
    VectorMax3d vlist[8];
    VectorMax3d widthratio;
    bool box_in_[3];
    bool check[3];
    bool use_ms;
    bool check_t_overlap;
    bool use_skip;
    bool this_level_less_tol;
    bool find_level_root;
    bool zero_in;
    bool box_in;
    bool ck0; 
    bool ck1;
    bool ck2;
    bool tol_condition;
    bool condition1;
    bool condition2;
    bool condition3;
    bool inserted;
    
    int refine;
    int current_level; // in the begining, current_level != level
    int box_in_level;
    int level;
    int split_i;
    Numccd low_number(0, 0);
    Numccd up_number(1, 0);
    Numccd TOI_SKIP;
    Numccd TOI;
    Singleinterval init_interval(low_number, up_number);
    Singleinterval iset[3];
    Singleinterval current[3];
    Singleinterval itv0, itv1, itv2;
    Singleinterval bisect_inter;
    interval_pair halves;
    MinHeap istack;
    item current_item;


    overflow_flag = 0;
    
    compute_face_vertex_tolerance_3d_new(data_in, co_domain_tolerance, tol);


#ifdef TIME_UPPER_IS_ONE
    check_t_overlap = false; // if input max_time = 1, then no need to check overlap
#else
    check_t_overlap = true;
#endif

    
    iset[0] = init_interval;
    iset[1] = init_interval;
    iset[2] = init_interval;

    // interval_root_finder_double_horizontal_tree
    overflow_flag = NO_OVERFLOW;
    output_tolerance = co_domain_tolerance;

    // this is used to catch the tolerance for each level
    temp_output_tolerance = co_domain_tolerance;

    // current intervals
    

    // LINENBR 2
    refine = 0;

    toi = SCALAR_LIMIT; //set toi as infinate
    // temp_toi is to catch the toi of each level
    temp_toi = toi;
    
    TOI.first = 4;
    TOI.second = 0; // set TOI as 4. this is to record the impact time of this level
    TOI_SKIP = TOI;               // this is to record the element that already small enough or contained in eps-box
    use_skip = false; // this is to record if TOI_SKIP is used.

    current_level = -2; // in the begining, current_level != level
    box_in_level = -2;  // this checks if all the boxes before this
    // level < tolerance. only true, we can return when we find one overlaps eps box and smaller than tolerance or eps-box
    this_level_less_tol = true;
    find_level_root = false;
    // Scalar current_tolerance=std::numeric_limits<Scalar>::infinity(); // set returned tolerance as infinite
    t_upper_bound = max_t; // 2*tol make it more conservative

    // LINENBR 3
    istack.insertKey(item(iset, -1));
    //return true;
    
    //LINENBR 5
    while (!istack.empty())
    {
        if (overflow_flag != NO_OVERFLOW)
        {
            break;
        }

        //LINENBR 6
        current_item = istack.extractMin();

        current[0] = current_item.itv[0];
        current[1] = current_item.itv[1];
        current[2] = current_item.itv[2];
        level = current_item.level;

        // if this box is later than TOI_SKIP in time, we can skip this one.
        // TOI_SKIP is only updated when the box is small enough or totally contained in eps-box
        if (!less_than(current[0].first, TOI_SKIP))
        {
            continue;
        }
        if (box_in_level != level)
        { // before check a new level, set this_level_less_tol=true
            box_in_level = level;
            this_level_less_tol = true;
        }
        // LINENBR 8
        refine++;

        box_in = false;
        zero_in = false;

        // zero_in =
        //     Origin_in_function_bounding_box_double_vector_return_tolerance(
        //         current, a0s, a1s, b0s, b1s, a0e, a1e, b0e, b1e, check_vf,
        //         err, ms, box_in, true_tol);
        //return zero_in;// REGSCOUNT 100
        
        if (!zero_in)
            continue;

        VectorMax3d widths = width(current);

        tol_condition = true_tol[0] <= co_domain_tolerance && true_tol[1] <= co_domain_tolerance && true_tol[2] <= co_domain_tolerance;

        // Condition 1, stopping condition on t, u and v is satisfied. this is useless now since we have condition 2
        condition1 = widths.v[0] <= tol[0] && widths.v[1] <= tol[1] && widths.v[2] <= tol[2];

        // Condition 2, zero_in = true, box inside eps-box and in this level,
        // no box whose zero_in is true but box size larger than tolerance, can return
        condition2 = box_in && this_level_less_tol;
        if (!tol_condition)
        {
            this_level_less_tol = false;
            // this level has at least one box whose size > tolerance, thus we
            // cannot directly return if find one box whose size < tolerance or box-in
        }

        // Condition 3, in this level, we find a box that zero-in and size < tolerance.
        // and no other boxes whose zero-in is true in this level before this one is larger than tolerance, can return
        condition3 = this_level_less_tol;
        // LINENBR 15, 16
        if (condition1 || condition2 || condition3)
        {
            TOI = current[0].first;

            // continue;
            toi = Numccd2double(TOI);

            return true;
        }

        // LINENBR 10
        if (current_level != level)
        {
            // LINENBR 22
            current_level = level;
            find_level_root = false;
        }
        if (!find_level_root)
        {
            TOI = current[0].first;

            // LINENBR 11
            // continue;
            // this is the first toi of this level
            temp_toi = Numccd2double(TOI);

            // if the real tolerance is larger than input, use the real one;
            // if the real tolerance is smaller than input, use input
            temp_output_tolerance = max(
                max(
                    max(true_tol[0], true_tol[1]), true_tol[2]),
                co_domain_tolerance);

            find_level_root =
                true; // this ensures always find the earlist root
        }

        // LINENBR 12
        if (refine > max_itr)
        {
            overflow_flag = ITERATION_OVERFLOW;
            break;
        }

        // if this box is small enough, or inside of eps-box, then just continue,
        // but we need to record the collision time
        if (tol_condition || box_in)
        {
            if (less_than(current[0].first, TOI_SKIP))
            {
                TOI_SKIP = current[0].first;
            }
            use_skip = true;
            continue;
        }

        

        check[0] = false;
        check[1] = false;
        check[2] = false;
        for (int i = 0; i < 3; i++)
        {
            widthratio.v[i] = widths.v[i] / tol[i];
            if (widths.v[i] > tol[i])
                check[i] = true; // means this need to be checked
        }

        split_i = -1;

        for (int i = 0; i < 3; i++)
        {
            if (check[i])
            {
                if (check[(i + 1) % 3] && check[(i + 2) % 3])
                {
                    if (widthratio.v[i] >= widthratio.v[(i + 1) % 3] && widthratio.v[i] >= widthratio.v[(i + 2) % 3])
                    {
                        split_i = i;
                        break;
                    }
                }
                if (check[(i + 1) % 3] && !check[(i + 2) % 3])
                {
                    if (widthratio.v[i] >= widthratio.v[(i + 1) % 3])
                    {
                        split_i = i;
                        break;
                    }
                }
                if (!check[(i + 1) % 3] && check[(i + 2) % 3])
                {
                    if (widthratio.v[i] >= widthratio.v[(i + 2) % 3])
                    {
                        split_i = i;
                        break;
                    }
                }
                if (!check[(i + 1) % 3] && !check[(i + 2) % 3])
                {

                    split_i = i;
                    break;
                }
            }
        }

        
        bisect_inter = current[split_i];
        // LINENBR 19
        bisect(bisect_inter, halves);
        if (!less_than(halves.first.first, halves.first.second))
        {
            overflow_flag = BISECTION_OVERFLOW;
            break;
        }
        if (!less_than(halves.second.first, halves.second.second))
        {
            overflow_flag = BISECTION_OVERFLOW;
            break;
        }
#ifndef CHECK_EE // check vf

        if (split_i == 1)
        {

            if (sum_no_larger_1(halves.second.first, current[2].first))
            {

                current[split_i] = halves.second;
                // LINENBR 20
                inserted = istack.insertKey(item(current, level + 1));
                if (inserted == false)
                {
                    overflow_flag = HEAP_OVERFLOW;
                }
            }
            if (sum_no_larger_1(halves.first.first, current[2].first))
            {
                current[split_i] = halves.first;
                inserted = istack.insertKey(item(current, level + 1));
                if (inserted == false)
                {
                    overflow_flag = HEAP_OVERFLOW;
                }
            }
        }

        if (split_i == 2)
        {

            if (sum_no_larger_1(halves.second.first, current[1].first))
            {
                current[split_i] = halves.second;
                inserted = istack.insertKey(item(current, level + 1));
                if (inserted == false)
                {
                    overflow_flag = HEAP_OVERFLOW;
                }
            }
            if (sum_no_larger_1(halves.first.first, current[1].first))
            {
                current[split_i] = halves.first;
                inserted = istack.insertKey(item(current, level + 1));
                if (inserted == false)
                {
                    overflow_flag = HEAP_OVERFLOW;
                }
            }
        }
        if (split_i == 0)
        {
            if (check_t_overlap)
            {
                if (interval_overlap_region(
                        halves.second, 0, t_upper_bound))
                {
                    current[split_i] = halves.second;
                    inserted = istack.insertKey(item(current, level + 1));
                    if (inserted == false)
                    {
                        overflow_flag = HEAP_OVERFLOW;
                    }
                }
                if (interval_overlap_region(
                        halves.first, 0, t_upper_bound))
                {
                    current[split_i] = halves.first;
                    inserted = istack.insertKey(item(current, level + 1));
                    if (inserted == false)
                    {
                        overflow_flag = HEAP_OVERFLOW;
                    }
                }
            }
            else
            {
                current[split_i] = halves.second;
                inserted = istack.insertKey(item(current, level + 1));
                if (inserted == false)
                {
                    overflow_flag = HEAP_OVERFLOW;
                }
                current[split_i] = halves.first;
                inserted = istack.insertKey(item(current, level + 1));
                if (inserted == false)
                {
                    overflow_flag = HEAP_OVERFLOW;
                }
            }
        }

#else
        if (check_t_overlap && split_i == 0)
        {
            if (interval_overlap_region(
                    halves.second, 0, t_upper_bound))
            {
                current[split_i] = halves.second;
                inserted = istack.insertKey(item(current, level + 1));
                if (inserted == false)
                {
                    overflow_flag = HEAP_OVERFLOW;
                }
            }
            if (interval_overlap_region(halves.first, 0, t_upper_bound))
            {
                current[split_i] = halves.first;
                inserted = istack.insertKey(item(current, level + 1));
                if (inserted == false)
                {
                    overflow_flag = HEAP_OVERFLOW;
                }
            }
        }
        else
        {
            current[split_i] = halves.second;
            inserted = istack.insertKey(item(current, level + 1));
            if (inserted == false)
            {
                overflow_flag = HEAP_OVERFLOW;
            }
            current[split_i] = halves.first;
            inserted = istack.insertKey(item(current, level + 1));
            if (inserted == false)
            {
                overflow_flag = HEAP_OVERFLOW;
            }
        }
#endif
    }

    if (overflow_flag > 0)
    {
        toi = temp_toi;
        output_tolerance = temp_output_tolerance;
        return true;
    }

    if (use_skip)
    {
        toi = Numccd2double(TOI_SKIP);

        return true;
    }
    return false;
}
__device__ bool edgeEdgeCCD_double(
    const CCDdata &data_in,
    const Scalar *err,
    const Scalar ms,
    Scalar &toi,
    Scalar tolerance,
    Scalar t_max,
    const int max_itr,
    Scalar &output_tolerance,
    bool no_zero_toi,
    int &overflow_flag)
{
    overflow_flag = 0;

    bool is_impacting;

    Scalar tol[3];

    Scalar err1[3];

    compute_edge_edge_tolerance_new(data_in, tolerance, tol);

    //////////////////////////////////////////////////////////

#ifdef CALCULATE_ERROR_BOUND
    VectorMax3d vlist[8];
#pragma unroll
    for (int i = 0; i < 3; i++)
    {
        vlist[0].v[i] = data_in.v0s[i];
        vlist[1].v[i] = data_in.v1s[i];
        vlist[2].v[i] = data_in.v2s[i];
        vlist[3].v[i] = data_in.v3s[i];
        vlist[4].v[i] = data_in.v0e[i];
        vlist[5].v[i] = data_in.v1e[i];
        vlist[6].v[i] = data_in.v2e[i];
        vlist[7].v[i] = data_in.v3e[i];
    }

    bool use_ms = ms > 0;
    get_numerical_error(vlist, 8, /*is_vf*/ false, use_ms, err1);
#else
    err1[0] = err[0];
    err1[1] = err[1];
    err1[2] = err[2];
#endif

#ifdef TIME_UPPER_IS_ONE
    bool check_t_overlap = false; // if input max_time = 1, then no need to check overlap
#else
    bool check_t_overlap = true;
#endif

    const Numccd low_number(0, 0);
    const Numccd up_number(1, 0);
    const Singleinterval init_interval(low_number, up_number);
    Singleinterval iset[3];
    iset[0] = init_interval;
    iset[1] = init_interval;
    iset[2] = init_interval;

    is_impacting = interval_root_finder_double_horizontal_tree(
        tol, tolerance, iset, check_t_overlap, t_max, toi,
        /*is_vf*/ false, err1, ms, data_in.v0s, data_in.v1s, data_in.v2s, data_in.v3s,
        data_in.v0e, data_in.v1e, data_in.v2e, data_in.v3e, max_itr,
        output_tolerance, overflow_flag);

    if (overflow_flag)
    {
        return true;
    }

    return is_impacting;
}

//
