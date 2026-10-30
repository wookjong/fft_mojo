"""Select grouped owners and boundary IO from untouched canonical stage IR."""
from dataclasses import replace
from planning.execution.fft_leaf_ownership import LeafOwnershipPlan
from planning.execution.fft_boundary_simd import make_boundary_simd_plan, extract_scalar_stages
from planning.strategies.fft_plan_recursive import FFTRecursiveNodePlan, FFTLeafPlan, flatten_recursive_node


def make_grouped_boundary_plan(plan,*,mode,near_group,far_group):
    from planning.execution.fft_boundary_fusion import FFTBoundaryFusionPlan,BoundaryBufferBinding
    def unsupported(reason):return FFTBoundaryFusionPlan(mode,False,reason)
    root=plan.root
    if (mode not in ('off','pre','post','pre-post') or not isinstance(root,FFTRecursiveNodePlan)
            or not isinstance(root.near_fft,FFTLeafPlan) or not isinstance(root.far_child,FFTLeafPlan)):
        return unsupported('grouped IO supports off/pre/post/pre-post and one recursive level')
    if near_group not in (1,4) or far_group not in (1,2,4) or (near_group,far_group)==(1,1):
        return unsupported('grouped IO requires near C1/C4 and far C1/C2/C4, at least one grouped')
    if ('pre' in mode and near_group!=4) or ('post' in mode and far_group==1):
        return unsupported('a fused boundary requires its own grouped leaf ownership')
    semantic=tuple(flatten_recursive_node(root))
    if len(semantic)!=5:return unsupported('grouped IO needs five semantic nodes')
    nodes=list(semantic)
    try:
        for index,c,boundary,m in ((1,near_group,'pre',root.a),(3,far_group,'post',root.b)):
            k=nodes[index]
            extract_scalar_stages(k)
            if c==1:continue
            ownership=LeafOwnershipPlan(m,k.length,c,plan.batch,k.distributed.num_units,
                k.distributed.workers_per_unit,tuple(x.name for x in k.workspace_buffers))
            special=(make_boundary_simd_plan(k,boundary=boundary,a=root.a,b=root.b,
                total_batch=plan.batch,group_leaves=c,window_batch=plan.batch) if boundary in mode else None)
            nodes[index]=replace(k,leaf_ownership=ownership,boundary_simd=special)
    except ValueError as error:return unsupported(str(error))
    removed=tuple(i for i,name in ((0,'pre'),(4,'post')) if name in mode)
    ids=tuple(i for i in range(5) if i not in removed)
    bindings=tuple(BoundaryBufferBinding(s,'input' if p==0 else f'work{(p-1)%2}',
        'output' if p==len(ids)-1 else f'work{p%2}') for p,s in enumerate(ids))
    return FFTBoundaryFusionPlan(mode,True,'grouped ownership and explicit cross-leaf SIMD',
        tuple(nodes[i] for i in ids),bindings,removed,None,1.0/plan.n if plan.inverse and 'post' in mode else None)
