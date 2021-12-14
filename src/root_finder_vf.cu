#include <gputi/root_finder.h>
#include <gputi/queue.h>
#include <iostream>
#include <float.h>
#include <vector>
#include <array>
namespace ccd
{
    CCDdata array_to_ccd(std::array<std::array<Scalar, 3>, 8> a)
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

    __device__ Singleinterval::Singleinterval(const Scalar &f, const Scalar &s)
    {
        first = f;
        second = s;
    }

    // this function do the bisection
    __device__ interval_pair::interval_pair(const Singleinterval &itv)
    {
        Scalar c = (itv.first + itv.second) / 2;
        first.first = itv.first;
        first.second = c;
        second.first = c;
        second.second = itv.second;
    }

    // t1+t2<=1?
    // true, when t1 + t2 < 1 / (1 + DBL_EPSILON);
    // false, when  t1 + t2 > 1 / (1 - DBL_EPSILON);
    // unknow, otherwise.
    __device__ bool sum_no_larger_1(const Scalar &num1, const Scalar &num2)
    {
#ifdef GPUTI_USE_DOUBLE_PRECISION
        if (num1 + num2 > 1 / (1 - DBL_EPSILON))
        {
            return false;
        }
#else
        if (num1 + num2 > 1 / (1 - FLT_EPSILON))
        {
            return false;
        }
#endif
        return true;
    }

    __device__ void compute_face_vertex_tolerance(const CCDdata &data_in, const CCDConfig &config, CCDOut &out)
    {
        Scalar p000[3], p001[3], p011[3], p010[3], p100[3], p101[3], p111[3], p110[3];
        for (int i = 0; i < 3; i++)
        {
            p000[i] = data_in.v0s[i] - data_in.v1s[i];
            p001[i] = data_in.v0s[i] - data_in.v3s[i];
            p011[i] = data_in.v0s[i] - (data_in.v2s[i] + data_in.v3s[i] - data_in.v1s[i]);
            p010[i] = data_in.v0s[i] - data_in.v2s[i];
            p100[i] = data_in.v0e[i] - data_in.v1e[i];
            p101[i] = data_in.v0e[i] - data_in.v3e[i];
            p111[i] = data_in.v0e[i] - (data_in.v2e[i] + data_in.v3e[i] - data_in.v1e[i]);
            p110[i] = data_in.v0e[i] - data_in.v2e[i];
        }
        Scalar dl = 0;
        for (int i = 0; i < 3; i++)
        {
            dl = max(dl, fabs(p100[i] - p000[i]));
            dl = max(dl, fabs(p101[i] - p001[i]));
            dl = max(dl, fabs(p111[i] - p011[i]));
            dl = max(dl, fabs(p110[i] - p010[i]));
        }
        dl *= 3;
        out.tol[0] = config.co_domain_tolerance / dl;

        dl = 0;
        for (int i = 0; i < 3; i++)
        {
            dl = max(dl, fabs(p010[i] - p000[i]));
            dl = max(dl, fabs(p110[i] - p100[i]));
            dl = max(dl, fabs(p111[i] - p101[i]));
            dl = max(dl, fabs(p011[i] - p001[i]));
        }
        dl *= 3;
        out.tol[1] = config.co_domain_tolerance / dl;

        dl = 0;
        for (int i = 0; i < 3; i++)
        {
            dl = max(dl, fabs(p001[i] - p000[i]));
            dl = max(dl, fabs(p101[i] - p100[i]));
            dl = max(dl, fabs(p111[i] - p110[i]));
            dl = max(dl, fabs(p011[i] - p010[i]));
        }
        dl *= 3;
        out.tol[2] = config.co_domain_tolerance / dl;
    }
    __device__ void compute_face_vertex_tolerance_memory_pool(CCDdata &data_in, const CCDConfig &config)
    {
        Scalar p000[3], p001[3], p011[3], p010[3], p100[3], p101[3], p111[3], p110[3];
        for (int i = 0; i < 3; i++)
        {
            p000[i] = data_in.v0s[i] - data_in.v1s[i];
            p001[i] = data_in.v0s[i] - data_in.v3s[i];
            p011[i] = data_in.v0s[i] - (data_in.v2s[i] + data_in.v3s[i] - data_in.v1s[i]);
            p010[i] = data_in.v0s[i] - data_in.v2s[i];
            p100[i] = data_in.v0e[i] - data_in.v1e[i];
            p101[i] = data_in.v0e[i] - data_in.v3e[i];
            p111[i] = data_in.v0e[i] - (data_in.v2e[i] + data_in.v3e[i] - data_in.v1e[i]);
            p110[i] = data_in.v0e[i] - data_in.v2e[i];
        }
        Scalar dl = 0;
        for (int i = 0; i < 3; i++)
        {
            dl = max(dl, fabs(p100[i] - p000[i]));
            dl = max(dl, fabs(p101[i] - p001[i]));
            dl = max(dl, fabs(p111[i] - p011[i]));
            dl = max(dl, fabs(p110[i] - p010[i]));
        }
        dl *= 3;
        data_in.tol[0] = config.co_domain_tolerance / dl;

        dl = 0;
        for (int i = 0; i < 3; i++)
        {
            dl = max(dl, fabs(p010[i] - p000[i]));
            dl = max(dl, fabs(p110[i] - p100[i]));
            dl = max(dl, fabs(p111[i] - p101[i]));
            dl = max(dl, fabs(p011[i] - p001[i]));
        }
        dl *= 3;
        data_in.tol[1] = config.co_domain_tolerance / dl;

        dl = 0;
        for (int i = 0; i < 3; i++)
        {
            dl = max(dl, fabs(p001[i] - p000[i]));
            dl = max(dl, fabs(p101[i] - p100[i]));
            dl = max(dl, fabs(p111[i] - p110[i]));
            dl = max(dl, fabs(p011[i] - p010[i]));
        }
        dl *= 3;
        data_in.tol[2] = config.co_domain_tolerance / dl;
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

        xmax = max(xmax, fabs(data_in.v1s[0]));
        ymax = max(ymax, fabs(data_in.v1s[1]));
        zmax = max(zmax, fabs(data_in.v1s[2]));

        xmax = max(xmax, fabs(data_in.v2s[0]));
        ymax = max(ymax, fabs(data_in.v2s[1]));
        zmax = max(zmax, fabs(data_in.v2s[2]));

        xmax = max(xmax, fabs(data_in.v3s[0]));
        ymax = max(ymax, fabs(data_in.v3s[1]));
        zmax = max(zmax, fabs(data_in.v3s[2]));

        xmax = max(xmax, fabs(data_in.v0e[0]));
        ymax = max(ymax, fabs(data_in.v0e[1]));
        zmax = max(zmax, fabs(data_in.v0e[2]));

        xmax = max(xmax, fabs(data_in.v1e[0]));
        ymax = max(ymax, fabs(data_in.v1e[1]));
        zmax = max(zmax, fabs(data_in.v1e[2]));

        xmax = max(xmax, fabs(data_in.v2e[0]));
        ymax = max(ymax, fabs(data_in.v2e[1]));
        zmax = max(zmax, fabs(data_in.v2e[2]));

        xmax = max(xmax, fabs(data_in.v3e[0]));
        ymax = max(ymax, fabs(data_in.v3e[1]));
        zmax = max(zmax, fabs(data_in.v3e[2]));

        xmax = max(xmax, Scalar(1));
        ymax = max(ymax, Scalar(1));
        zmax = max(zmax, Scalar(1));

        box.err[0] = xmax * xmax * xmax * vffilter;
        box.err[1] = ymax * ymax * ymax * vffilter;
        box.err[2] = zmax * zmax * zmax * vffilter;
        return;
    }
    __device__ __host__ void get_numerical_error_vf_memory_pool(
        CCDdata &data_in)
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

        xmax = max(xmax, fabs(data_in.v1s[0]));
        ymax = max(ymax, fabs(data_in.v1s[1]));
        zmax = max(zmax, fabs(data_in.v1s[2]));

        xmax = max(xmax, fabs(data_in.v2s[0]));
        ymax = max(ymax, fabs(data_in.v2s[1]));
        zmax = max(zmax, fabs(data_in.v2s[2]));

        xmax = max(xmax, fabs(data_in.v3s[0]));
        ymax = max(ymax, fabs(data_in.v3s[1]));
        zmax = max(zmax, fabs(data_in.v3s[2]));

        xmax = max(xmax, fabs(data_in.v0e[0]));
        ymax = max(ymax, fabs(data_in.v0e[1]));
        zmax = max(zmax, fabs(data_in.v0e[2]));

        xmax = max(xmax, fabs(data_in.v1e[0]));
        ymax = max(ymax, fabs(data_in.v1e[1]));
        zmax = max(zmax, fabs(data_in.v1e[2]));

        xmax = max(xmax, fabs(data_in.v2e[0]));
        ymax = max(ymax, fabs(data_in.v2e[1]));
        zmax = max(zmax, fabs(data_in.v2e[2]));

        xmax = max(xmax, fabs(data_in.v3e[0]));
        ymax = max(ymax, fabs(data_in.v3e[1]));
        zmax = max(zmax, fabs(data_in.v3e[2]));

        xmax = max(xmax, Scalar(1));
        ymax = max(ymax, Scalar(1));
        zmax = max(zmax, Scalar(1));

        data_in.err[0] = xmax * xmax * xmax * vffilter;
        data_in.err[1] = ymax * ymax * ymax * vffilter;
        data_in.err[2] = zmax * zmax * zmax * vffilter;
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
    __device__ void BoxPrimatives::calculate_tuv(const BoxCompute &box)
    {
        if (b[0] == 0)
        { // t0
            t = box.current_item.itv[0].first;
        }
        else
        { // t1
            t = box.current_item.itv[0].second;
        }

        if (b[1] == 0)
        { // u0
            u = box.current_item.itv[1].first;
        }
        else
        { // u1
            u = box.current_item.itv[1].second;
        }

        if (b[2] == 0)
        { // v0
            v = box.current_item.itv[2].first;
        }
        else
        { // v1
            v = box.current_item.itv[2].second;
        }
    }
    __device__ void BoxPrimatives::calculate_tuv(const MP_unit &unit)
    {
        if (b[0] == 0)
        { // t0
            t = unit.itv[0].first;
        }
        else
        { // t1
            t = unit.itv[0].second;
        }

        if (b[1] == 0)
        { // u0
            u = unit.itv[1].first;
        }
        else
        { // u1
            u = unit.itv[1].second;
        }

        if (b[2] == 0)
        { // v0
            v = unit.itv[2].first;
        }
        else
        { // v1
            v = unit.itv[2].second;
        }
    }
    __device__ Scalar calculate_vf(const CCDdata &data_in, const BoxPrimatives &bp)
    {
        Scalar v, pt, t0, t1, t2;
        v = (data_in.v0e[bp.dim] - data_in.v0s[bp.dim]) * bp.t + data_in.v0s[bp.dim];
        t0 = (data_in.v1e[bp.dim] - data_in.v1s[bp.dim]) * bp.t + data_in.v1s[bp.dim];
        t1 = (data_in.v2e[bp.dim] - data_in.v2s[bp.dim]) * bp.t + data_in.v2s[bp.dim];
        t2 = (data_in.v3e[bp.dim] - data_in.v3s[bp.dim]) * bp.t + data_in.v3s[bp.dim];
        pt = (t1 - t0) * bp.u + (t2 - t0) * bp.v + t0;
        return (v - pt);
    }

    __device__ bool Origin_in_vf_inclusion_function(const CCDdata &data_in, BoxCompute &box, CCDOut &out)
    {
        BoxPrimatives bp;
        Scalar vmin = SCALAR_LIMIT;
        Scalar vmax = -SCALAR_LIMIT;
        Scalar value;
        for (bp.dim = 0; bp.dim < 3; bp.dim++)
        {
            vmin = SCALAR_LIMIT;
            vmax = -SCALAR_LIMIT;
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
    __device__ bool Origin_in_vf_inclusion_function_memory_pool(const CCDdata &data_in, MP_unit &unit)
    {
        BoxPrimatives bp;
        Scalar vmin = SCALAR_LIMIT;
        Scalar vmax = -SCALAR_LIMIT;
        Scalar value;
        for (bp.dim = 0; bp.dim < 3; bp.dim++)
        {
            vmin = SCALAR_LIMIT;
            vmax = -SCALAR_LIMIT;
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    for (int k = 0; k < 2; k++)
                    {
                        bp.b[0] = i;
                        bp.b[1] = j;
                        bp.b[2] = k; //100
                        bp.calculate_tuv(unit);
                        value = calculate_vf(data_in, bp);
                        vmin = min(vmin, value);
                        vmax = max(vmax, value);
                    }
                }
            }

            // get the min and max in one dimension
            unit.true_tol = max(unit.true_tol, vmax - vmin); // this is the real tolerance

            if (vmin > data_in.err[bp.dim] || vmax < -data_in.err[bp.dim])
            {
                return false;
            }

            if (vmin < -data_in.err[bp.dim] || vmax > data_in.err[bp.dim])
            {
                unit.box_in = false;
            }
        }
        return true;
    }
    __device__ void split_dimension(const CCDOut &out, BoxCompute &box)
    { // clarified in queue.h
        Scalar res[3];
        res[0] = box.widths[0] / out.tol[0];
        res[1] = box.widths[1] / out.tol[1];
        res[2] = box.widths[2] / out.tol[2];
        if (res[0] >= res[1] && res[0] >= res[2])
        {
            box.split = 0;
        }
        if (res[1] >= res[0] && res[1] >= res[2])
        {
            box.split = 1;
        }
        if (res[2] >= res[1] && res[2] >= res[0])
        {
            box.split = 2;
        }
    }

    __device__ void bisect_vf_and_push(BoxCompute &box, const CCDConfig &config, MinHeap &istack, CCDOut &out)
    {
        interval_pair halves(box.current_item.itv[box.split]); // bisected
        bool inserted;
        if (halves.first.first >= halves.first.second)
        {
            out.overflow_flag = BISECTION_OVERFLOW;
            return;
        }
        if (halves.second.first >= halves.second.second)
        {
            out.overflow_flag = BISECTION_OVERFLOW;
            return;
        }
        if (box.split == 0) // split t interval
        {
            if (config.max_t != 1)
            {
                if (halves.second.first <= config.max_t)
                {
                    box.current_item.itv[box.split] = halves.second;
                    inserted = istack.insertKey(box.current_item.itv, box.current_item.level + 1);
                    if (inserted == false)
                    {
                        out.overflow_flag = HEAP_OVERFLOW;
                    }
                }

                box.current_item.itv[box.split] = halves.first;
                inserted = istack.insertKey(box.current_item.itv, box.current_item.level + 1);
                if (inserted == false)
                {
                    out.overflow_flag = HEAP_OVERFLOW;
                }
            }
            else
            {
                box.current_item.itv[box.split] = halves.second;
                inserted = istack.insertKey(box.current_item.itv, box.current_item.level + 1);
                if (inserted == false)
                {
                    out.overflow_flag = HEAP_OVERFLOW;
                }
                box.current_item.itv[box.split] = halves.first;
                inserted = istack.insertKey(box.current_item.itv, box.current_item.level + 1);
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
                inserted = istack.insertKey(box.current_item.itv, box.current_item.level + 1);
                if (inserted == false)
                {
                    out.overflow_flag = HEAP_OVERFLOW;
                }
            }

            box.current_item.itv[box.split] = halves.first;
            inserted = istack.insertKey(box.current_item.itv, box.current_item.level + 1);
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
                inserted = istack.insertKey(box.current_item.itv, box.current_item.level + 1);
                if (inserted == false)
                {
                    out.overflow_flag = HEAP_OVERFLOW;
                }
            }

            box.current_item.itv[box.split] = halves.first;
            inserted = istack.insertKey(box.current_item.itv, box.current_item.level + 1);
            if (inserted == false)
            {
                out.overflow_flag = HEAP_OVERFLOW;
            }
        }
    }

    __device__ void vertexFaceCCD(const CCDdata &data_in, const CCDConfig &config, CCDOut &out)
    {

        MinHeap istack; // now when initialized, size is 1 and initialized with [0,1]^3
        compute_face_vertex_tolerance(data_in, config, out);
        BoxCompute box;

#ifdef CALCULATE_ERROR_BOUND
        get_numerical_error_vf(data_in, box);
#else
        box.err[0] = config.err_in[0];
        box.err[1] = config.err_in[1];
        box.err[2] = config.err_in[2];
#endif

        // LINENBR 2
        int refine = 0;
        bool zero_in;
        bool condition;
        // level < tolerance. only true, we can return when we find one overlaps eps box and smaller than tolerance or eps-box

        while (!istack.empty())
        {
            //LINENBR 6
            istack.extractMin(box.current_item); // get the level and the intervals
            // LINENBR 8
            refine++;
            zero_in =
                Origin_in_vf_inclusion_function(data_in, box, out);

            if (!zero_in)
                continue;

            // get the width of the box
            box.widths[0] = box.current_item.itv[0].second - box.current_item.itv[0].first;
            box.widths[1] = box.current_item.itv[1].second - box.current_item.itv[1].first;
            box.widths[2] = box.current_item.itv[2].second - box.current_item.itv[2].first;

            // LINENBR 15, 16
            // Condition 1, if the tolerance is smaller than the threadshold, return true;
            condition = box.widths[0] <= out.tol[0] && box.widths[1] <= out.tol[1] && box.widths[2] <= out.tol[2];
            if (condition)
            {
                out.result = true;
                return;
            }
            // Condition 2, the box is inside the epsilon box, have a root, return true;
            condition = box.box_in;
            if (condition)
            {
                out.result = true;
                return;
            }

            // Condition 3, real tolerance is smaller than the input tolerance, return true
            condition = box.true_tol <= config.co_domain_tolerance;
            if (condition)
            {
                out.result = true;
                return;
            }

            // LINENBR 12
            if (refine > config.max_itr)
            {
                out.overflow_flag = ITERATION_OVERFLOW;
                out.result = true;
                return;
            }
            split_dimension(out, box);
            bisect_vf_and_push(box, config, istack, out);
            if (out.overflow_flag != NO_OVERFLOW)
            {
                out.result = true;
                return;
            }
        }
        out.result = false;
        return;
    }

    __global__ void run_parallel_memory_pool_vf_ccd_all(CCDdata *data, CCDConfig *config_in, bool *res, int size, Scalar *tois)
    {
        int tx = threadIdx.x + blockIdx.x * blockDim.x;
        if (tx >= size)
            return;
        // copy the input queries to __device__
        CCDdata data_in;
        for (int i = 0; i < 3; i++)
        {
            data_in.v0s[i] = data[tx].v0s[i];
            data_in.v1s[i] = data[tx].v1s[i];
            data_in.v2s[i] = data[tx].v2s[i];
            data_in.v3s[i] = data[tx].v3s[i];
            data_in.v0e[i] = data[tx].v0e[i];
            data_in.v1e[i] = data[tx].v1e[i];
            data_in.v2e[i] = data[tx].v2e[i];
            data_in.v3e[i] = data[tx].v3e[i];
        }
        // copy the configurations to the shared memory
        __shared__ CCDConfig config;
        config.err_in[0] = config_in->err_in[0];
        config.err_in[1] = config_in->err_in[1];
        config.err_in[2] = config_in->err_in[2];
        config.co_domain_tolerance = config_in->co_domain_tolerance; // tolerance of the co-domain
        config.max_t = config_in->max_t;                             // the upper bound of the time interval
        config.max_itr = config_in->max_itr;                         // the maximal nbr of iterations
        CCDOut out;
        vertexFaceCCD(data_in, config, out);
        res[tx] = out.result;
        tois[tx] = 0;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // the memory pool method
    __global__ void compute_vf_tolerance_memory_pool(CCDdata *data, CCDConfig *config, const int query_size)
    {
        int tx = threadIdx.x + blockIdx.x * blockDim.x;
        if (tx >= query_size)
            return;
        compute_face_vertex_tolerance_memory_pool(data[tx], config[0]);

        data[tx].last_round_has_root = false;
        data[tx].last_round_has_root_record = false;
        data[tx].sure_have_root = false;
        data[tx].nbr_checks = 0;
#ifdef CALCULATE_ERROR_BOUND
        get_numerical_error_vf_memory_pool(data[tx]);
#endif
        __syncthreads();
    }
    // the size of units is UNIT_SIZE;
    __global__ void initialize_memory_pool(MP_unit *units, int query_size)
    {
        int tx = threadIdx.x + blockIdx.x * blockDim.x;
        if (tx >= UNIT_SIZE)
            return;
        if (tx >= query_size)
        {
            units[tx].init(-1);
        }
        else
        {
            units[tx].init(tx);
        }

        __syncthreads();
    }
    __device__ void split_dimension_memory_pool(const CCDdata &data, Scalar width[3], int &split)
    { // clarified in queue.h
        Scalar res[3];
        res[0] = width[0] / data.tol[0];
        res[1] = width[1] / data.tol[1];
        res[2] = width[2] / data.tol[2];
        if (res[0] >= res[1] && res[0] >= res[2])
        {
            split = 0;
        }
        if (res[1] >= res[0] && res[1] >= res[2])
        {
            split = 1;
        }
        if (res[2] >= res[1] && res[2] >= res[0])
        {
            split = 2;
        }
    }

    __device__ void bisect_vf_memory_pool(MP_unit &unit, int split, const CCDConfig &config, MP_unit bisected[2], int &valid_nbr)
    {
        interval_pair halves(unit.itv[split]); // bisected

        if (halves.first.first >= halves.first.second)
        {
            valid_nbr = 0;
            return;
        }
        if (halves.second.first >= halves.second.second)
        {
            valid_nbr = 0;
            return;
        }
        bisected[0] = unit;
        bisected[1] = unit;
        valid_nbr = 1;
        bisected[0].itv[split] = halves.first;

        if (split == 0)
        {
            if (config.max_t != 1)
            {
                if (halves.second.first <= config.max_t)
                {
                    bisected[1].itv[split] = halves.second;
                    valid_nbr = 2;
                }
            }
            else
            {
                bisected[1].itv[split] = halves.second;
                valid_nbr = 2;
            }
        }
        if (split == 1)
        {
            if (sum_no_larger_1(halves.second.first, bisected[1].itv[2].first)) // check if u+v<=1
            {

                bisected[1].itv[1] = halves.second;
                valid_nbr = 2;
            }
        }
        if (split == 2)
        {
            if (sum_no_larger_1(halves.second.first, bisected[1].itv[1].first)) // check if u+v<=1
            {

                bisected[1].itv[2] = halves.second;
                valid_nbr = 2;
            }
        }
    }
    __global__ void vf_ccd_memory_pool(MP_unit *units, int query_size, CCDdata *data, CCDConfig *config, int *results)
    {
        int tx = threadIdx.x + blockIdx.x * blockDim.x;
        if (tx >= UNIT_SIZE)
            return;

        // int start;
        // int end;
        // bool mp_status=true;//meaning that the start<=end;
        // start=0;
        // end=query_size-1;
        int refine = 0;
        Scalar widths[3];
        bool condition;
        int split;
        bool break_thread = false; // to record if the for loop should break
        for (int i = 0; i < 100; i++)
        {
            bool no_need_check = false;
            int box_id = units[tx].query_id;
            atomicExch(&config[0].not_empty, 0); // mark this level as empty elements
            if (box_id == -1)
            { // if this box indices no query, no need to check;
                no_need_check = true;
            }
            else
            {
                if (refine > 0 && data[box_id].last_round_has_root == 0)
                {                         // if this query does not have root in the last step, no need to check
                    no_need_check = true; // TODO think about it
                }
                if (data[box_id].sure_have_root > 0)
                { // if it is sure that have root, then no need to check
                    no_need_check = true;
                }
            }
            units[tx].query_id = -1; // this unit will be regarded as empty in the next iteration
            __syncthreads();
            if (tx < query_size)
            {
                data[tx].last_round_has_root_record = data[tx].last_round_has_root;
                data[tx].last_round_has_root = 0;
            }
            __syncthreads();
            if (!no_need_check && !break_thread)
            {                                       // if need check, and the for loop is not broken, do the check
                atomicAdd(&config[0].not_empty, 1); // if find at least one box that is not empty, not_empty is true;
                bool zero_in = Origin_in_vf_inclusion_function_memory_pool(data[box_id], units[tx]);
                if (zero_in)
                {
                    widths[0] = units[tx].itv[0].second - units[tx].itv[0].first;
                    widths[1] = units[tx].itv[1].second - units[tx].itv[1].first;
                    widths[2] = units[tx].itv[2].second - units[tx].itv[2].first;

                    // Condition 1
                    condition = widths[0] <= data[box_id].tol[0] && widths[1] <= data[box_id].tol[1] && widths[2] <= data[box_id].tol[2];
                    if (condition)
                    {
                        atomicAdd(&data[box_id].sure_have_root, 1);
                    }
                    // Condition 2, the box is inside the epsilon box, have a root, return true;
                    condition = units[tx].box_in;
                    if (condition)
                    {
                        atomicAdd(&data[box_id].sure_have_root, 1);
                    }

                    // Condition 3, real tolerance is smaller than the input tolerance, return true
                    condition = units[tx].true_tol <= config->co_domain_tolerance;
                    if (condition)
                    {
                        atomicAdd(&data[box_id].sure_have_root, 1);
                    }
                    atomicAdd(&data[box_id].last_round_has_root, 1); // if contain origin, then last_round_has_root +=1
                    split_dimension_memory_pool(data[box_id], widths, split);
                    MP_unit bisected[2];
                    int valid_nbr;
                    bisect_vf_memory_pool(units[tx], split, config[0], bisected, valid_nbr);

                    bisected[0].query_id = box_id;
                    bisected[1].query_id = box_id;
                    if (valid_nbr == 0)
                    { // in this case, the interval is too small that overflow happens
                        atomicAdd(&data[box_id].sure_have_root, 1);
                    }
                    if (valid_nbr == 1)
                    {
                        int unit_id = atomicInc(&config[0].mp_end, (unsigned int)UNIT_SIZE); // this is the new id of the new box. end >= UNIT_SIZE? 0 : (end+1)
                        if (units[(unit_id + 1) % UNIT_SIZE].query_id != -1)
                        { // it means it is not empty
                            atomicAdd(&config[0].mp_status, 1);
                        }
                        units[(unit_id + 1) % UNIT_SIZE] = bisected[0];
                    }
                    if (valid_nbr == 2)
                    {
                        int unit_id = atomicInc(&config[0].mp_end, UNIT_SIZE); // this is the new id of the new box
                        if (units[(unit_id + 1) % UNIT_SIZE].query_id != -1)
                        {
                            atomicAdd(&config[0].mp_status, 1);
                        }
                        units[(unit_id + 1) % UNIT_SIZE] = bisected[0];

                        unit_id = atomicInc(&config[0].mp_end, UNIT_SIZE); // this is the new id of the new box
                        if (units[(unit_id + 1) % UNIT_SIZE].query_id != -1)
                        {
                            atomicAdd(&config[0].mp_status, 1);
                        }
                        units[(unit_id + 1) % UNIT_SIZE] = bisected[1];
                    }
                }
            }

            refine++;
            __syncthreads();

            if (config[0].not_empty == 0)
            { // if all the boxes are checked, return;

                break_thread = true;
            }
            if (refine > HEAP_SIZE || config[0].mp_status > 0)
            { // if check too many times or exceed the size of the heap, return results according to the last step
                if (tx < query_size)
                {
                    if (data[tx].last_round_has_root_record > 0)
                    {
                        data[tx].sure_have_root = 1;
                    }
                }

                break_thread = true;
            }

            __syncthreads();
            //atomicAdd(data[box_id], 1);

            // TODO remember to set the last_round_has_root as 0 or 1 after synchronization.
            // TODO deal with config.mp_start, modify it globally.
            // TODO check if the queue size is sufficient, if not, biset the data set. (maybe add a variable "uncertain", in addition to sure_have_root. and
            // "current_data_set_size" and "need_to_bisect_data_set" and "queue_size_of_this_query")
        }
        //__syncthreads();
        if (tx < query_size)
        {
            if (data[tx].sure_have_root > 0)
            {
                results[tx] = 1; // get the results
            }
            else
            {
                results[tx] = 0;
            }
        }

        //
    }

    void run_memory_pool_ccd(const std::vector<std::array<std::array<Scalar, 3>, 8>> &V, bool is_edge,
                             std::vector<bool> &result_list, int parallel_nbr, double &runtime)
    {
        int nbr = V.size();
        result_list.resize(nbr);
        // host
        CCDdata *data_list = new CCDdata[nbr];
        for (int i = 0; i < nbr; i++)
        {
            data_list[i] = array_to_ccd(V[i]);
#ifndef NO_CHECK_MS
            data_list[i].ms = MINIMUM_SEPARATION_BENCHMARK;
#endif
        }

        int *res = new int[nbr];
        MP_unit *units = new MP_unit[UNIT_SIZE];
        CCDConfig *config = new CCDConfig[1];
        config[0].err_in[0] = -1;             // the input error bound calculate from the AABB of the whole mesh
        config[0].co_domain_tolerance = 1e-6; // tolerance of the co-domain
        config[0].max_t = 1;                  // the upper bound of the time interval
        config[0].max_itr = 1e6;              // the maximal nbr of iterations
        config[0].mp_end = nbr - 1;           // the initialized trunk is from 0 to nbr-1;
        config[0].mp_status = true;           // in the begining, start < end
        config[0].not_empty = 0;
        // device
        CCDdata *d_data_list;
        int *d_res;
        MP_unit *d_units;
        CCDConfig *d_config;

        int data_size = sizeof(CCDdata) * nbr;
        int result_size = sizeof(int) * nbr;
        int unit_size = sizeof(MP_unit) * UNIT_SIZE;
        // int dbg_size=sizeof(Scalar)*8;

        cudaMalloc(&d_data_list, data_size);
        cudaMalloc(&d_res, result_size);
        cudaMalloc(&d_units, unit_size);
        cudaMalloc(&d_config, sizeof(CCDConfig));

        cudaMemcpy(d_data_list, data_list, data_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_config, config, sizeof(CCDConfig), cudaMemcpyHostToDevice);
        // all the memory copied. now firstly initialize the memory pool
        ccd::Timer timer;
        timer.start();
        initialize_memory_pool<<<UNIT_SIZE / parallel_nbr + 1, parallel_nbr>>>(d_units, nbr);
        cudaDeviceSynchronize();
        compute_vf_tolerance_memory_pool<<<UNIT_SIZE / parallel_nbr + 1, parallel_nbr>>>(d_data_list, d_config, nbr);
        cudaDeviceSynchronize();
        vf_ccd_memory_pool<<<UNIT_SIZE / parallel_nbr + 1, parallel_nbr>>>(d_units, nbr, d_data_list, d_config, d_res);
        cudaDeviceSynchronize();
        double tt = timer.getElapsedTimeInMicroSec();
        runtime = tt;

        cudaDeviceSynchronize();

        cudaProfilerStop();

        cudaMemcpy(res, d_res, result_size, cudaMemcpyDeviceToHost);
        //cudaMemcpy(tois, d_tois, time_size, cudaMemcpyDeviceToHost);
        //cudaMemcpy(dbg, d_dbg, dbg_size, cudaMemcpyDeviceToHost);

        cudaFree(d_data_list);
        cudaFree(d_res);
        cudaFree(d_units);
        cudaFree(d_config);
        //cudaFree(d_dbg);

        for (int i = 0; i < nbr; i++)
        {
            result_list[i] = res[i];
        }

        // std::cout << "dbg info\n"
        //           << dbg[0] << "," << dbg[1] << "," << dbg[2] << "," << dbg[3] << "," << dbg[4] << "," << dbg[5] << "," << dbg[6] << "," << dbg[7] << std::endl;
        std::cout << "dbg result " << res[0] << std::endl;
        delete[] res;
        delete[] data_list;
        delete[] units;
        delete[] config;
        cudaError_t ct = cudaGetLastError();
        printf("******************\n%s\n************\n", cudaGetErrorString(ct));

        return;
    }
}
