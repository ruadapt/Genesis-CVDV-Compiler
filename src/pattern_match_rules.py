from src.pattern_match import *

def _is_const(node:ParsedNode):
    if node.name in ('nsign', 'sum', 'prod', 'complex'):
        return all(_is_const(c) for c in node.children)
    try:
        complex(node.name)
        return True
    except ValueError:
        return False

def __is_real_const(node:ParsedNode):
    if node.name in ('nsign', 'sum', 'prod'):
        return all(_is_const(c) for c in node.children)
    try:
        float(node.name)
        return True
    except ValueError:
        return False

def __is_squared_factor(node:ParsedNode|complex, unit_scalar:complex):
    try:
        if isinstance(node, ParsedNode):
            if node.children != []:
                return False
            z = complex(node.name)
        else:
            z = node
        z2 = z/unit_scalar
        if z2.imag != 0 or z2.real < 0:
            return False
        return True
    except ValueError:
        return False

def __get_squared_factor(node:ParsedNode|complex, unit_scalar:complex):
    if isinstance(node, ParsedNode):
        z = complex(node.name)
    else:
        z = node
    z2 = z/unit_scalar
    if z2.imag != 0:
        raise Exception(f"{z} -> {z2} cannot be squared")
    return math.sqrt(z2.real)

def __check_equality_list(vars:dict[str, ParsedNode], conditions_list:list[tuple[ParsedNode, ParsedNode]], reduction_rules:list[Rule]=None):
    prepared_conds = [(c1.copy(), c2.copy()) for c1,c2 in conditions_list]
    for vname,val in vars.items():
        for c1, c2 in prepared_conds:
            c1.replace_all(ParsedNode(vname), val)
            c2.replace_all(ParsedNode(vname), val)
    
    if reduction_rules == None:
        reduction_rules = basic_rules_list

    for c1, c2 in prepared_conds:
        #print(f"comparing: {c1}, {c2}")
        nc1 = [x[1] for x in apply_rules_list_greedy(c1, reduction_rules)]
        nc2 = [x[1] for x in apply_rules_list_greedy(c2, reduction_rules)]
        if any(o1 == o2 for o1 in nc1 for o2 in nc2):
            #print("compare passed")
            continue
        else:
            return False
    return True

def __increment_counter(env:dict, key:str):
    if key not in env.keys():
        env[key] = 0
    out = env[key]
    env[key] += 1
    return ParsedNode(str(out)) 

def __is_mode(node:ParsedNode):
    try:
        int(node.name)
        return False
    except ValueError:
        pass
    if node.name == '1' or node.name in assoc_ops:
        return False
    if node.children == []:
        return True
    if len(node.children) == 1:
        try:
            int(node.children[0].name)
            return True
        except ValueError:
            return False
    return False

terminate_patterns = [
    #1
    (parse_str('BOD(1)'),
     [],
    ),
    #(parse_str('prod(1)'),
    # [],
    #),
]


commute_rules_list = [
    Rule(parse_str('sum(prod(?c, ?1,?2),prod(?nc,?2,?1)))'), ['?1', '?2', '?c', '?nc'], parse_str('prod(?c, comm(?1, ?2))'),
         lambda x : not __check_equality_list(x, [(parse_str("?1"), parse_str("?2"))]) and
        __check_equality_list(x, [(parse_str("?c"), parse_str("nsign(?nc)"))])
         
         ),
    Rule(parse_str('sum(prod(?1,?2),prod(?2,?1))'), ['?1', '?2'], parse_str('acomm(?1, ?2)')),
    #(parse_str('comm(?1, ?1)'), ['?1'], parse_str('0')),
    #(parse_str('acomm(?1, ?1)'), ['?1'], parse_str('prod(2, ?1)')),
    #(parse_str('comm(?1, ?2)'), ['?1', '?2'], parse_str('sum(prod(?1,?2),nsign(prod(?2,?1)))')),
    #(parse_str('acomm(?1, ?2)'), ['?1', '?2'], parse_str('sum(prod(?1,?2),prod(?2,?1))')),
]

gate_cancel_rules = [
    Rule(parse_str('prod(qgate(h, qb1), qgate(h, qb1))'), ['qb1'], parse_str('prod')),
    Rule(parse_str('prod(qgate(x, qb1), qgate(x, qb1))'), ['qb1'], parse_str('prod')),
    Rule(parse_str('prod(qgate(s, qb1), qgate(dagger(s), qb1))'), ['qb1'], parse_str('prod')),
    Rule(parse_str('prod(qgate(dagger(s), qb1), qgate(s, qb1))'), ['qb1'], parse_str('prod')),
]

basic_rules_list = [
    Rule(parse_str('dagger(tprod(?1, ?2))'), ['?1', '?2'], parse_str('tprod(dagger(?2), dagger(?1)))'),
     lambda x : x['?1'].name != '1' and x['?2'].name != '1'),
    Rule(parse_str('dagger(prod(?t, ?1, ?2))'), ['?1', '?2', '?t'], parse_str('prod(?t, dagger(?2), dagger(?1)))'),
     lambda x : x['?1'].name != '1' and x['?2'].name != '1' and _is_const(x['?t'])),
    Rule(parse_str('dagger(prod(?1, ?2))'), ['?1', '?2'], parse_str('prod(dagger(?2), dagger(?1)))'),
     lambda x : x['?1'].name != '1' and x['?2'].name != '1'),
    Rule(parse_str('dagger(dagger(?1))'), ['?1'], parse_str('?1')),
    Rule(parse_str('prod(-1,-1,?1)'), ['?1'], parse_str('?1')),
    Rule(parse_str('sum(prod(?2,?1), prod(?3,?1))'), ['?1','?2','?3'], parse_str('0'),
         lambda x : __check_equality_list(x, [(parse_str('?2'), parse_str('nsign(?3)'))], [])  
         ),
    Rule(parse_str('sum(prod(?2,?1), prod(?3,?1))'), ['?1','?2','?3'], parse_str('prod(?4,?1)'),
         lambda x : _is_const(x['?2']) and _is_const(x['?3']) and
         not __check_equality_list(x, [(parse_str('?2'), parse_str('nsign(?3)'))], []) ,
         {'?4': (lambda x,_: complex(x['?2'].name) + complex(x['?3'].name))}
         ),
    #Rule(parse_str('prod(bc(?1), ba(?2))'), ['?1', '?2'], parse_str('1')),
    Rule(parse_str('square(?1)'), ['?1'], parse_str('prod(?1, ?1)')),
    Rule(parse_str('exp(0)'), [], parse_str('1')),

    #Rule(parse_str('conj(sum(?1, prod(j,?2)))'), ['?1', '?2'], parse_str('sum(?1, prod(-j,?2))'), lambda x: __is_real_const(x['?1']) and __is_real_const(x['?2'])),
    #Rule(parse_str('conj(sum(?1, nsign(prod(j,?2))))'), ['?1', '?2'], parse_str('sum(?1, prod(j,?2))'), lambda x: __is_real_const(x['?1']) and __is_real_const(x['?2'])),
    #Rule(parse_str('conj(sum(prod(j,?1), ?2))'), ['?1', '?2'], parse_str('sum(nsign(prod(j,?1)), ?2)'), lambda x: __is_real_const(x['?1']) and __is_real_const(x['?2'])),
    #Rule(parse_str('conj(sum(nsign(prod(j,?1)), ?2))'), ['?1', '?2'], parse_str('sum(prod(j,?1), ?2)'), lambda x: __is_real_const(x['?1']) and __is_real_const(x['?2'])),
    #Rule(parse_str('conj(?1)'), ['?1'], parse_str('?1'), lambda x: __is_real_const(x['?1'])),

    Rule(parse_str('prod(j, j)'), [], parse_str('-1')),
    Rule(parse_str('prod(0, ?1)'), ['?1'], parse_str('0')),
    Rule(parse_str('prod(?1, 0)'), ['?1'], parse_str('0')),
    Rule(parse_str('dagger(qgate(s, qb1))'), ['qb1'], parse_str('qgate(sdg, qb1)')),
    Rule(parse_str('dagger(qgate(sdg, qb1))'), ['qb1'], parse_str('qgate(s, qb1)')),
    Rule(parse_str('qgate(dagger(x), qb1)'), ['qb1'], parse_str('qgate(x, qb1)')),
    Rule(parse_str('square(x)'), [], parse_str('prod(x,x)')),

]
exp_misc_rules = [
    #13 This is specifically for gates with operator senario
    Rule(
        parse_str('exp( prod(t, N, M) )'),
        ['M', 'N', 't'],
        parse_str('prod( N, exp(prod(t, M)))'),
        lambda x : _is_const(x['t']) and x['N'].name == 'qgate'
    ),
    Rule(
        parse_str('exp( prod(M, N) )'),
        ['M', 'N'],
        parse_str('prod( exp(M), N)'),
        lambda x : x['N'].name == 'qgate'
    ), 

]

branching_rules_list = [
    Rule(parse_str('prod(w, sum(prod(x, y), prod(x, z)))'), ['w', 'x', 'y', 'z'], parse_str('prod(v, sum(prod(1, y), prod(1, z)))'), 
         lambda x: x['x'].name != '1', 
         {'v':(lambda x,_: ParsedNode('prod', [x['w'].copy(), x['x'].copy()]))}),
    Rule(parse_str('prod(w, sum(prod(x, y), prod(x2, z)))'), ['w', 'x', 'x2', 'y', 'z'], parse_str('prod(v, sum(prod(1, y), prod(-1, z)))'), 
         lambda x: x['x'].name != '1' and __check_equality_list(x, [(parse_str('x'), parse_str('nsign(x2)'))]), 
         {'v':(lambda x,_: ParsedNode('prod', [x['w'].copy(), x['x'].copy()]))}),
    Rule(parse_str('sum(prod(x, y), prod(x, z))'), ['x', 'y', 'z'], parse_str('prod(x, sum(prod(1, y), prod(1, z)))'), 
         lambda x: x['x'].name != '1'),
]

decomp_rules_list = [

    #distribute exp over sum
    Rule(parse_str('exp(sum(?1, ?2))'), ['?1', '?2'], parse_str('prod(exp(?1), exp(?2))'), 
        lambda x: x['?1'] != parse_str('0') and x['?2'] != parse_str('0') and (x['?1'].name != 'sum' or (len(x['?1'].children) == 2))
    ),

    #3
    Rule(parse_str('exp( t, comm(M, N)) )'),
      ['M', 'N', 't'],
      parse_str('BCH(prod(t2, sigma(3, new_i), N), prod(t2, sigma(3, new_i), M))'),
      lambda x : (
          __check_equality_list(x, [(parse_str('M'), parse_str('dagger(M)')), (parse_str('N'), parse_str('dagger(N)'))])
            and __is_squared_factor(x['t'], 1)
            ),
    {'t2':(lambda x,_ : __get_squared_factor(x['t'], 1) * 1j), 
     't3':(lambda x,_ : __get_squared_factor(x['t'], 1) * -1j),
     'new_i':(lambda _,env : __increment_counter(env.index_counters, 'qubit'))
    }),
    Rule(parse_str('exp( t, comm(M, N)) )'),
      ['M', 'N', 't'],
      parse_str('BCH(prod(t3, sigma(3, new_i), N), prod(t2, sigma(3, new_i), M))'),
      lambda x : (
          __check_equality_list(x, [(parse_str('M'), parse_str('dagger(M)')), (parse_str('N'), parse_str('dagger(N)'))])
            and __is_squared_factor(x['t'], -1)
            ),
    {'t2':(lambda x,_ : __get_squared_factor(x['t'], -1) * 1j),
     't3':(lambda x,_ : __get_squared_factor(x['t'], -1) * -1j),
     'new_i':(lambda _,env : __increment_counter(env.index_counters, 'qubit'))
    }),
    

    #4 
    Rule(parse_str('exp(prod(t, sigma(3, qb1), acomm(M, N) ))'),
      ['M', 'N', 't', 'qb1'],
      parse_str('BCH( prod(t2, sigma(1, qb1), M), prod(t2, sigma(2, qb1), N) )'),
      lambda x : (__check_equality_list(x, [(parse_str('M'), parse_str('dagger(M)')), (parse_str('N'), parse_str('dagger(N)'))]) and
            __is_squared_factor(x['t'], -1j)
      ),
      dict(t2=lambda x,_ : __get_squared_factor(x['t'], -1j) * 1j)),
    Rule(parse_str('exp(prod(t, sigma(2, qb1), acomm(M, N) ))'),
      ['M', 'N', 't', 'qb1'],
      parse_str('BCH( prod(t2, sigma(3, qb1), M), prod(t2, sigma(1, qb1), N) )'),
      lambda x : (__check_equality_list(x, [(parse_str('M'), parse_str('dagger(M)')), (parse_str('N'), parse_str('dagger(N)'))]) and
            __is_squared_factor(x['t'], -1j)
      ),
      dict(t2=lambda x,_ : __get_squared_factor(x['t'], -1j) * 1j)),
    Rule(parse_str('exp(prod(t, sigma(1, qb1), acomm(M, N) ))'),
      ['M', 'N', 't', 'qb1'],
      parse_str('BCH( prod(t2, sigma(2, qb1), M), prod(t2, sigma(3, qb1), N) )'),
      lambda x : (__check_equality_list(x, [(parse_str('M'), parse_str('dagger(M)')), (parse_str('N'), parse_str('dagger(N)'))]) and
            __is_squared_factor(x['t'], -1j)
      ),
      dict(t2=lambda x,_ : __get_squared_factor(x['t'], -1j) * 1j)),
    
    Rule(parse_str('exp(prod(t, sigma(3, qb1), acomm(M, N) ))'),
      ['M', 'N', 't', 'qb1'],
      parse_str('BCH( prod(nsign(t2), sigma(1, qb1), M), prod(t2, sigma(2, qb1), N) )'),
      lambda x : (__check_equality_list(x, [(parse_str('M'), parse_str('dagger(M)')), (parse_str('N'), parse_str('dagger(N)'))]) and
            __is_squared_factor(x['t'], 1j)
      ),
      dict(t2=lambda x,_ : __get_squared_factor(x['t'], 1j) * 1j)),
    Rule(parse_str('exp(prod(t, sigma(2, qb1), acomm(M, N) ))'),
      ['M', 'N', 't', 'qb1'],
      parse_str('BCH( prod(nsign(t2), sigma(3, qb1), M), prod(t2, sigma(1, qb1), N) )'),
      lambda x : (__check_equality_list(x, [(parse_str('M'), parse_str('dagger(M)')), (parse_str('N'), parse_str('dagger(N)'))]) and
            __is_squared_factor(x['t'], 1j)
      ),
      dict(t2=lambda x,_ : __get_squared_factor(x['t'], 1j) * 1j)),
    Rule(parse_str('exp(prod(t, sigma(1, qb1), acomm(M, N) ))'),
      ['M', 'N', 't', 'qb1'],
      parse_str('BCH( prod(nsign(t2), sigma(2, qb1), M), prod(t2, sigma(3, qb1), N) )'),
      lambda x : (__check_equality_list(x, [(parse_str('M'), parse_str('dagger(M)')), (parse_str('N'), parse_str('dagger(N)'))]) and
            __is_squared_factor(x['t'], 1j)
      ),
      dict(t2=lambda x,_ : __get_squared_factor(x['t'], 1j) * 1j)),

    #5
    Rule(parse_str('exp(prod(t, sigma(3, qb1), comm(M, N)))'),
      ['M', 'N', 't', 'qb1'],
      parse_str('BCH( prod(t2, N),  prod(t2, sigma(3, qb1), M))'),
      lambda x : ( __is_squared_factor(x['t'], -1j)),
      {'t2':(lambda x,_ : __get_squared_factor(x['t'], -1j) * 1j)}
      ),
    Rule(parse_str('exp(prod(t, sigma(3, qb1), comm(M, N)))'),
      ['M', 'N', 't', 'qb1'],
      parse_str('BCH( prod(nsign(t2), N),  prod(t2, sigma(3, qb1), M))'),
      lambda x : ( __is_squared_factor(x['t'], 1j)),
      {'t2':(lambda x,_ : __get_squared_factor(x['t'], 1j) * 1j)}
      ),

    #6
    Rule(parse_str('exp(prod(t, sigma(3, qb1), sum(prod(c1,M,N), prod(c2, Ndg, Mdg)))))'),
      ['M', 'N', 'Mdg', 'Ndg', 't', 'c1', 'c2', 'qb1'],
      parse_str('BCH( prod(t2, qgate(x, qb1), BOD(N, qb2), qgate(x, qb1)), prod(t2, BOD(M, qb3)))'),
      lambda x : x['M'].name != '1' and x['N'].name != '1' and __is_squared_factor(complex(x['t'].name)*complex(x['c1'].name), 1) 
        and __check_equality_list(x, [
          (parse_str('sum(prod(1,M,N), prod(-1,N,M))'), parse_str('0')),
          (parse_str('dagger(M)'), parse_str('Mdg')),
          (parse_str('dagger(N)'), parse_str('Ndg')),
          (parse_str('c1'), parse_str('nsign(c2)')),
        ]),
    dict(t2=lambda x,_ : __get_squared_factor(complex(x['t'].name)*complex(x['c1'].name), 1) * 1j,
         qb2=(lambda _,env : __increment_counter(env.index_counters, 'qubit')),
         qb3=(lambda _,env : __increment_counter(env.index_counters, 'qubit'))
         )),
    #6b
    Rule(parse_str('exp(prod(t, sigma(3, qb1), sum(prod(c1,M,N), prod(c2, Ndg, Mdg)))))'),
      ['M', 'N', 'Mdg', 'Ndg', 't', 'c1', 'c2', 'qb1'],
      parse_str('BCH( prod(nsign(t2), qgate(x, qb1), BOD(N, qb2), qgate(x, qb1)), prod(t2, BOD(M, qb3)))'),
      lambda x : x['M'].name != '1' and x['N'].name != '1' and __is_squared_factor(complex(x['t'].name)*complex(x['c1'].name), -1) 
        and __check_equality_list(x, [
          (parse_str('sum(prod(1,M,N), prod(-1,N,M))'), parse_str('0')),
          (parse_str('dagger(M)'), parse_str('Mdg')),
          (parse_str('dagger(N)'), parse_str('Ndg')),
          (parse_str('c1'), parse_str('nsign(c2)')),
        ]),
    dict(t2=lambda x,_ : __get_squared_factor(complex(x['t'].name)*complex(x['c1'].name), -1) * 1j,
         qb2=(lambda _,env : __increment_counter(env.index_counters, 'qubit')),
         qb3=(lambda _,env : __increment_counter(env.index_counters, 'qubit'))
         )),
    
    #7
    Rule(parse_str('exp(prod(t, sigma(3, qb1), sum(prod(c1,M,N), prod(c1, Ndg, Mdg)))))'),
      ['M', 'N', 'Mdg', 'Ndg', 't', 'c1', 'qb1'],
      parse_str('BCH( prod(t2, qgate(s, qb1), BOD(N, qb2), qgate(sdg, qb1)), prod(t2, qgate(x, qb1), BOD(M, qb3), qgate(x, qb1)))'),
      lambda x : x['M'].name != '1' and x['N'].name != '1' and __is_squared_factor(complex(x['t'].name)*complex(x['c1'].name), 1j) 
        and __check_equality_list(x, [
          (parse_str('sum(prod(1,M,N), prod(-1,N,M))'), parse_str('0')),
          (parse_str('dagger(M)'), parse_str('Mdg')),
          (parse_str('dagger(N)'), parse_str('Ndg')),
        ]),
    dict(t2=lambda x,_ : __get_squared_factor(complex(x['t'].name)*complex(x['c1'].name), 1j) * 1j,
         qb2=(lambda _,env : __increment_counter(env.index_counters, 'qubit')),
         qb3=(lambda _,env : __increment_counter(env.index_counters, 'qubit'))
         )),
    #7b
    Rule(parse_str('exp(prod(t, sigma(3, qb1), sum(prod(c1,M,N), prod(c1, Ndg, Mdg)))))'),
      ['M', 'N', 'Mdg', 'Ndg', 't', 'c1', 'qb1'],
      parse_str('BCH( prod(nsign(t2), qgate(s, qb1), BOD(N, qb2), qgate(sdg, qb1)), prod(t2, qgate(x, qb1), BOD(M, qb3), qgate(x, qb1)))'),
      lambda x : x['M'].name != '1' and x['N'].name != '1' and __is_squared_factor(complex(x['t'].name)*complex(x['c1'].name), -1j) 
        and __check_equality_list(x, [
          (parse_str('sum(prod(1,M,N), prod(-1,N,M))'), parse_str('0')),
          (parse_str('dagger(M)'), parse_str('Mdg')),
          (parse_str('dagger(N)'), parse_str('Ndg')),
        ]),
    dict(t2=lambda x,_ : __get_squared_factor(complex(x['t'].name)*complex(x['c1'].name), -1j) * 1j,
         qb2=(lambda _,env : __increment_counter(env.index_counters, 'qubit')),
         qb3=(lambda _,env : __increment_counter(env.index_counters, 'qubit'))
         )),
    
    #8
    Rule(parse_str('exp(prod(t, M, N))'),
      ['t', 'M', 'N'],
      parse_str('trotter(prod(t2, sigma(3, qb1), comm(M,N)), prod(t2, sigma(3, qb1), acomm(M,N)))'),
      lambda x : _is_const(x['t']) and __check_equality_list(x, [(parse_str('M'), parse_str('dagger(M)')), (parse_str('N'), parse_str('dagger(N)'))]),
      {'t2':(lambda x,_ : complex(x['t'].name) / 2),
       'qb1':(lambda _,env : __increment_counter(env.index_counters, 'qubit'))}
      ),

    #9
    Rule(parse_str('exp(prod(t, M, N))'),
      ['M', 'N', 't'],
      parse_str('BCH( prod(t2, qgate(s, qb1), BOD(M, qb2), qgate(sdg, qb1)), prod(t2, qgate(x, qb1), BOD(N, qb3), qgate(x, qb1)))'),
      lambda x : __is_squared_factor(x['t'], 1j) and x['M'].name != '1' and x['N'].name != '1' and __check_equality_list(x, [(parse_str('sum(prod(1,M,N), prod(-1,N,M))'), parse_str('0')), (parse_str('prod(1,M,N)'), parse_str('dagger(prod(1,M,N))')) ]),
      dict(t2=lambda x,_ : __get_squared_factor(x['t'], 2j) * 1j,
         qb1=(lambda _,env : __increment_counter(env.index_counters, 'qubit')),
         qb2=(lambda _,env : __increment_counter(env.index_counters, 'qubit')),
         qb3=(lambda _,env : __increment_counter(env.index_counters, 'qubit'))
         )),
    Rule(parse_str('exp(prod(t, M, N))'),
      ['M', 'N', 't'],
      parse_str('BCH( prod(nsign(t2), qgate(s, qb1), BOD(M, qb2), qgate(sdg, qb1)), prod(t2, qgate(x, qb1), BOD(N, qb3), qgate(x, qb1)))'),
      lambda x : __is_squared_factor(x['t'], -1j) and x['M'].name != '1' and x['N'].name != '1' and __check_equality_list(x, [(parse_str('sum(prod(1,M,N), prod(-1,N,M))'), parse_str('0')), (parse_str('prod(1,M,N)'), parse_str('dagger(prod(1,M,N))')) ]),
      dict(t2=lambda x,_ : __get_squared_factor(x['t'], -2j) * 1j,
         qb1=(lambda _,env : __increment_counter(env.index_counters, 'qubit')),
         qb2=(lambda _,env : __increment_counter(env.index_counters, 'qubit')),
         qb3=(lambda _,env : __increment_counter(env.index_counters, 'qubit'))
         )),

    #10
    Rule(parse_str('exp(prod(t, BOD(prod(M,N), qb1)))'),
      ['M', 'N', 't', 'qb1'],
      parse_str('prod(qgate(x, qb1), trotter(prod(t2, sigma(2, qb1), sum(prod(1,M,N),nsign(dagger(prod(1,M,N))))),  prod(t3, sigma(1, qb1), sum(prod(1,M,N),dagger(prod(1,M,N))))), qgate(x, qb1) )'),
      lambda x : _is_const(x['t']) and x['M'] != parse_str('1') and x['N'] != parse_str('1') and
        __check_equality_list(x, [(parse_str('sum(prod(1, M,N), prod(-1, N,M)))'), parse_str('0')) ]),
      dict(t2=lambda x,_ : complex(x['t'].name)/(2j), t3=lambda x,_ : complex(x['t'].name)/(2))
    ),

    #11
    Rule(parse_str('exp( prod(t, M, N)))'),
      ['M', 'N', 't'],
      parse_str('BCH( prod(?t, qgate(s, qb1), BOD(M, qb2), qgate(sdg, qb1) ),  prod(?t, qgate(x, qb1), BOD(N, qb3), qgate(x, qb1) ) )'),
      lambda x: (x['M'].name != '1' and x['N'].name != '1' and
        __check_equality_list(x, [(parse_str('prod(1,M,N)'), parse_str('prod(1,dagger(N),dagger(M))'))]) and
        _is_const(x['t'])
      ),
      {'?t':(lambda x,_: complex(x['t'].name)/2),
         "qb1":(lambda _,env : __increment_counter(env.index_counters, 'qubit')),
         "qb2":(lambda _,env : __increment_counter(env.index_counters, 'qubit')),
         "qb3":(lambda _,env : __increment_counter(env.index_counters, 'qubit'))
    }),
]

def __make_seq(pstring:str, qmode:ParsedNode, rev:bool = False):
    out = ParsedNode('prod', [])
    for i, c in enumerate(pstring.strip()):
        if rev:
            rotation = [parse_str(f"qgate(CR({math.pi}), {i}, {str(qmode)})")]
        else:
            rotation = [parse_str(f"qgate(CR(-{math.pi}), {i}, {str(qmode)})")]


        if c == 'I':
            continue
        elif c == 'X':
            rotation_full = [parse_str(f"qgate(h, {i})")] + rotation + [parse_str(f"qgate(h, {i})")]
        elif c == 'Y':
            rotation_full = [parse_str(f"qgate(sdg, {i})"), parse_str(f"qgate(h, {i})")] + rotation + [parse_str(f"qgate(h, {i})"), parse_str(f"qgate(s, {i})")]
        elif c == 'Z':
            rotation_full = rotation
        else:
            raise Exception("Improper pauli string, please adjust rule conditions")
        
        if rev:
            out.children = rotation_full + out.children
        else:
            out.children.extend(rotation_full)
    return out

def __get_j_coeff(pstring:str):
    return (-1j) ** len([True for c in pstring.strip() if c != 'I'])

sigma_z_rules = [
    Rule(parse_str('exp(prod(?1, sigma(1, qb1), ?2))'), ['?1', '?2', 'qb1'], parse_str('prod(qgate(h, qb1), exp(prod(?1, sigma(3, qb1), ?2)), qgate(h, qb1))')),
    Rule(parse_str('exp(prod(?1, sigma(1, qb1)))'), ['?1', 'qb1'], parse_str('prod(qgate(h, qb1), exp(prod(?1, sigma(3, qb1))), qgate(h, qb1))')),
    Rule(parse_str('exp(prod(sigma(1, qb1), ?2))'), ['?2', 'qb1'], parse_str('prod(qgate(h, qb1), exp(prod(sigma(3, qb1) ?2)), qgate(h, qb1),)')),
    Rule(parse_str('exp(prod(sigma(1, qb1)))'), ['qb1'], parse_str('prod(qgate(h, qb1), exp(prod( sigma(3, qb1) )), qgate(h, qb1),)')),

    Rule(parse_str('exp(prod(?1, sigma(2, qb1), ?2))'), ['?1', '?2', 'qb1'], parse_str('prod(qgate(sdg, qb1), qgate(h, qb1), exp(prod(?1, sigma(3, qb1), ?2)), qgate(h, qb1), qgate(s, qb1),)')),
    Rule(parse_str('exp(prod(?1, sigma(2, qb1)))'), ['?1', 'qb1'], parse_str('prod(qgate(sdg, qb1), qgate(h, qb1), exp(prod(?1, sigma(3, qb1))), qgate(h, qb1), qgate(s, qb1),)')),
    Rule(parse_str('exp(prod(sigma(2, qb1), ?2))'), ['?2', 'qb1'], parse_str('prod(qgate(sdg, qb1), qgate(h, qb1), exp(prod(sigma(3, qb1) ?2)), qgate(h, qb1), qgate(s, qb1))')),
    Rule(parse_str('exp(prod(sigma(2, qb1)))'), ['qb1'], parse_str('prod(qgate(sdg, qb1), qgate(h, qb1), exp(prod( sigma(3, qb1) )), qgate(h, qb1), qgate(s, qb1))')),

    Rule(
        parse_str('exp(prod(?t, paulistring(?ps), sum(prod(?alpha, dagger(?1)), prod(?alphac, ?1))))'),
        ['?t', '?1', '?ps', '?alpha', '?alphac'],
        parse_str("prod(?revseq, qgate(D(prod(?t, ?alpha, ?im)), ?1), ?seq)"),
        lambda x : __is_mode(x['?1']) and __check_equality_list(x, [
            (parse_str('prod(?t, ?alpha)'), parse_str('nsign(conj(prod(?t, ?alphac)))')),
        ]),
        {'?revseq':(lambda x,_:__make_seq(x['?ps'].name, x['?1'], rev=True)),
         '?seq':(lambda x,_:__make_seq(x['?ps'].name, x['?1'])),
         '?im':(lambda x,_:__get_j_coeff(x['?ps'].name))}
    ),
    Rule(
        parse_str('exp(prod(?t, paulistring(?ps), sum(prod(?alpha, ?1), prod(?alphac, dagger(?1)))))'),
        ['?t', '?1', '?ps', '?alpha', '?alphac'],
        parse_str("prod(?revseq, qgate(D(prod(?t, ?alpha, ?im)), ?1), ?seq)"),
        lambda x : __is_mode(x['?1']) and __check_equality_list(x, [
            (parse_str('prod(?t, ?alpha)'), parse_str('nsign(conj(prod(?t, ?alphac)))')),
        ]),
        {'?revseq':(lambda x,_:__make_seq(x['?ps'].name, x['?1'], rev=True)),
         '?seq':(lambda x,_:__make_seq(x['?ps'].name, x['?1'])),
         '?im':(lambda x,_:__get_j_coeff(x['?ps'].name))}
    ),
]

basic_gates_list = [

    #Eliminate sigma(0)
    Rule(parse_str('sigma(0, qb1)'), ['qb1'], parse_str('1')),

    #1
    #"""Pauli"""
    #2
    Rule(
        parse_str('exp(prod(?angle, sigma(3, qb1)))'),
        ['?angle', 'qb1'],
        parse_str('qgate(rz(?newangle), qb1)'),
        lambda x : _is_const(x['?angle']),
        {'?newangle':(lambda x,_: complex(x['?angle'].name) * -2 / 1j)}
    ),
    #3
    Rule(  
        parse_str('exp(prod(?angle, dagger(?1), ?1))'),
        ['?1', '?angle'],
        parse_str('qgate(R(?newangle), ?1)'),
        lambda x : 
            (__is_mode(x['?1']) and _is_const(x['?angle'])),
        {'?newangle':(lambda x,_: complex(x['?angle'].name)/ -1j)}
    ),
    #4
    Rule(
        parse_str('exp(prod(?t, sum(prod(?alpha, dagger(?1)), prod(?alphac, ?1))))'),
        ['?1', '?t', '?alpha', '?alphac'],
        parse_str('qgate(D(?newalpha), ?1)'),
        lambda x : 
            (__is_mode(x['?1']) and _is_const(x['?t']) and _is_const(x['?alpha']) and
             __check_equality_list(x, [
                 (parse_str('prod(?t, ?alpha)'), parse_str('nsign(conj(prod(?t, ?alphac)))')),
             ])
             ),
        {'?newalpha':(lambda x,_: complex(x['?t'].name) * complex(x['?alpha'].name))}
    ),

    #4b
    Rule(
        parse_str('exp(prod(?t, sum(prod(?alpha, ?1), prod(?alphac, dagger(?1)))))'),
        ['?1', '?t', '?alpha', '?alphac'],
        parse_str('qgate(D(?newalpha), ?1)'),
        lambda x : 
            (__is_mode(x['?1']) and _is_const(x['?t']) and _is_const(x['?alpha']) and
             __check_equality_list(x, [
                 (parse_str('prod(?t, ?alpha)'), parse_str('nsign(conj(prod(?t, ?alphac)))')),
             ])
             ),
        {'?newalpha':(lambda x,_: complex(x['?t'].name) * complex(x['?alpha'].name))}
    ),

    #5
    Rule(
        parse_str('exp(prod(?angle, sum(prod(?cf1, dagger(?1), ?2), prod(?cf1, ?1, dagger(?2)))))'),
        ['?1', '?2', '?angle', '?cf1'],
        parse_str('qgate(BS(?newangle, 0), ?1, ?2)'),
        lambda x: 
            _is_const(x['?cf1']) and
            (__is_mode(x['?1'])) and
            (__is_mode(x['?2'])) and
            (_is_const(x['?angle'])) and
            not __check_equality_list(x, [
            (parse_str("?1"), parse_str("?2")),
          ]),
        {'?newangle':(lambda x,_: complex(x['?cf1'].name) * complex(x['?angle'].name) * 2 / -1j)}
    ),
    Rule(
        parse_str('exp(prod(?angle, sum(prod(?cf1, dagger(?1), ?2), prod(?cf1, dagger(?2), ?1))))'),
        ['?1', '?2', '?angle', '?cf1'],
        parse_str('qgate(BS(?newangle, 0), ?1, ?2)'),
        lambda x: 
            _is_const(x['?cf1']) and
            (__is_mode(x['?1'])) and
            (__is_mode(x['?2'])) and
            (_is_const(x['?angle'])) and
            not __check_equality_list(x, [
            (parse_str("?1"), parse_str("?2")),
          ]),
        {'?newangle':(lambda x,_: complex(x['?cf1'].name) * complex(x['?angle'].name) * 2 / -1j)}
    ),
    Rule(
        parse_str('exp(prod(?angle, sum(prod(?cf1, ?1, dagger(?2)), prod(?cf1, dagger(?1), ?2))))'),
        ['?1', '?2', '?angle', '?cf1'],
        parse_str('qgate(BS(?newangle, 0), ?1, ?2)'),
        lambda x: 
            _is_const(x['?cf1']) and
            (__is_mode(x['?1'])) and
            (__is_mode(x['?2'])) and
            (_is_const(x['?angle'])) and
            not __check_equality_list(x, [
            (parse_str("?1"), parse_str("?2")),
          ]),
        {'?newangle':(lambda x,_: complex(x['?cf1'].name) * complex(x['?angle'].name) * 2 / -1j)}
    ),
    Rule(
        parse_str('exp(prod(?angle, sum(prod(?cf1, ?1, dagger(?2)), prod(?cf1, ?2, dagger(?1)))))'),
        ['?1', '?2', '?angle', '?cf1'],
        parse_str('qgate(BS(?newangle, 0), ?1, ?2)'),
        lambda x: 
            _is_const(x['?cf1']) and
            (__is_mode(x['?1'])) and
            (__is_mode(x['?2'])) and
            (_is_const(x['?angle'])) and
            not __check_equality_list(x, [
            (parse_str("?1"), parse_str("?2")),
          ]),
        {'?newangle':(lambda x,_: complex(x['?cf1'].name) * complex(x['?angle'].name) * 2 / -1j)}
    ),
    
    #6
    Rule(  
        parse_str('exp(prod(?angle, sigma(3, qb1), dagger(?1), ?1))'),
        ['?1', '?angle', 'qb1'],
        parse_str('qgate(CR(?newangle), qb1, ?1)'),
        lambda x : 
            (__is_mode(x['?1']) and _is_const(x['?angle'])),
        {'?newangle':(lambda x,_: complex(x['?angle'].name) * 2 / -1j)}
    ),
    
    #7
    #"""Seems same as 6"""
    #8
    Rule(
        parse_str('exp(prod(?t, sigma(3, qb1), sum(prod(?alpha, dagger(?1)), prod(?alphac, ?1))))'),
        ['?1', '?t', 'qb1', '?alpha', '?alphac'],
        parse_str('qgate(CD(?newalpha), qb1, ?1)'),
        lambda x : 
            (__is_mode(x['?1']) and _is_const(x['?t']) and _is_const(x['?alpha']) and
             __check_equality_list(x, [
                 (parse_str('prod(?t, ?alpha)'), parse_str('nsign(conj(prod(?t, ?alphac)))')),
             ])
             ),
        {'?newalpha':(lambda x,_: complex(x['?t'].name) * complex(x['?alpha'].name))}
    ),
    Rule(
        parse_str('exp(prod(?t, sum(prod(?alpha, sigma(3, qb1), dagger(?1)), prod(?alphac, sigma(3, qb1), ?1))))'),
        ['?1', '?t', 'qb1', '?alpha', '?alphac'],
        parse_str('qgate(CD(?newalpha), qb1, ?1)'),
        lambda x : 
            (__is_mode(x['?1']) and _is_const(x['?t']) and _is_const(x['?alpha']) and (__is_squared_factor(x['?t'], 1) or __is_squared_factor(x['?t'], -1)) and
             __check_equality_list(x, [
                 (parse_str('?alpha'), parse_str('nsign(conj(?alphac))')),
             ])
             ),
        {'?newalpha':(lambda x,_: complex(x['?t'].name) * complex(x['?alpha'].name))}
    ),

    #8b
    Rule(
        parse_str('exp(prod(?t, sigma(3, qb1), sum(prod(?alpha, ?1), prod(?alphac, dagger(?1)))))'),
        ['?1', '?t', 'qb1', '?alpha', '?alphac'],
        parse_str('qgate(CD(?newalpha), qb1, ?1)'),
        lambda x : 
            (__is_mode(x['?1']) and _is_const(x['?t']) and _is_const(x['?alpha']) and
             __check_equality_list(x, [
                 (parse_str('prod(?t, ?alpha)'), parse_str('nsign(conj(prod(?t, ?alphac)))')),
             ])
             ),
        {'?newalpha':(lambda x,_: complex(x['?t'].name) * complex(x['?alpha'].name))}
    ),
    Rule(
        parse_str('exp(prod(?t, sum(prod(?alpha, sigma(3, qb1), ?1), prod(?alphac, sigma(3, qb1), dagger(?1)))))'),
        ['?1', '?t', 'qb1', '?alpha', '?alphac'],
        parse_str('qgate(CD(?newalpha), qb1, ?1)'),
        lambda x : 
            (__is_mode(x['?1']) and _is_const(x['?t']) and _is_const(x['?alpha']) and (__is_squared_factor(x['?t'], 1) or __is_squared_factor(x['?t'], -1)) and
             __check_equality_list(x, [
                 (parse_str('?alpha'), parse_str('nsign(conj(?alphac))')),
             ])
             ),
        {'?newalpha':(lambda x,_: complex(x['?t'].name) * complex(x['?alpha'].name))}
    ),

    #9
    Rule(
        parse_str('exp(prod(?angle, sigma(3, qb1), sum(prod(?cf1, dagger(?1), ?2), prod(?cf1, ?1, dagger(?2)))))'),
        ['?1', '?2', '?angle', '?cf1', 'qb1'],
        parse_str('qgate(CBS(?newangle, 0), qb1, ?1, ?2)'),
        lambda x: 
            _is_const(x['?cf1']) and
            (__is_mode(x['?1'])) and
            (__is_mode(x['?2'])) and
            (_is_const(x['?angle'])) and
            not __check_equality_list(x, [
            (parse_str("?1"), parse_str("?2")),
          ]),
        {'?newangle':(lambda x,_: complex(x['?cf1'].name) * complex(x['?angle'].name) * 2 / -1j)}
    ),
    Rule(
        parse_str('exp(prod(?angle, sigma(3, qb1), sum(prod(?cf1, dagger(?1), ?2), prod(?cf1, dagger(?2), ?1))))'),
        ['?1', '?2', '?angle', '?cf1', 'qb1'],
        parse_str('qgate(CBS(?newangle, 0), qb1, ?1, ?2)'),
        lambda x: 
            _is_const(x['?cf1']) and
            (__is_mode(x['?1'])) and
            (__is_mode(x['?2'])) and
            (_is_const(x['?angle'])) and
            not __check_equality_list(x, [
            (parse_str("?1"), parse_str("?2")),
          ]),
        {'?newangle':(lambda x,_: complex(x['?cf1'].name) * complex(x['?angle'].name) * 2 / -1j)}
    ),
    Rule(
        parse_str('exp(prod(?angle, sigma(3, qb1), sum(prod(?cf1, ?1, dagger(?2)), prod(?cf1, dagger(?1), ?2))))'),
        ['?1', '?2', '?angle', '?cf1', 'qb1'],
        parse_str('qgate(CBS(?newangle, 0), qb1, ?1, ?2)'),
        lambda x: 
            _is_const(x['?cf1']) and
            (__is_mode(x['?1'])) and
            (__is_mode(x['?2'])) and
            (_is_const(x['?angle'])) and
            not __check_equality_list(x, [
            (parse_str("?1"), parse_str("?2")),
          ]),
        {'?newangle':(lambda x,_: complex(x['?cf1'].name) * complex(x['?angle'].name) * 2 / -1j)}
    ),
    Rule(
        parse_str('exp(prod(?angle, sigma(3, qb1), sum(prod(?cf1, ?1, dagger(?2)), prod(?cf1, ?2, dagger(?1)))))'),
        ['?1', '?2', '?angle', '?cf1', 'qb1'],
        parse_str('qgate(CBS(?newangle, 0), qb1, ?1, ?2)'),
        lambda x: 
            _is_const(x['?cf1']) and
            (__is_mode(x['?1'])) and
            (__is_mode(x['?2'])) and
            (_is_const(x['?angle'])) and
            not __check_equality_list(x, [
            (parse_str("?1"), parse_str("?2")),
          ]),
        {'?newangle':(lambda x,_: complex(x['?cf1'].name) * complex(x['?angle'].name) * 2 / -1j)}
    ),
    
    #10
    Rule(
        parse_str('prod(?t, sigma(1, qb1), sum(prod(?angle, dagger(?1)), prod(?anglec, ?1)))'),
        ['?1', '?angle', '?anglec', 'qb1', '?t'],
        parse_str('qgate(RB(?newangle), qb1, ?1)'),
        lambda x : 
            (__is_mode(x['?1'])) and
            (__is_squared_factor(x['?t'], 1) or __is_squared_factor(x['?t'], -1)) and
            __check_equality_list(x, [
            (parse_str("?angle"), parse_str("conj(?anglec)")),
          ]),
        {'?newangle':(lambda x,_: complex(x['?t'].name) * complex(x['?angle'].name) -1j)}
    ),
    
    #Some decomp gates have been moved here

    #1
    Rule(parse_str('trotter(prod(t1, M), prod(t2, N))'),
     ['M', 'N', 't1', 't2'],
     parse_str('prod(exp(prod(t1, M)), exp(prod(t2, N)))'),
     lambda x : _is_const(x['t1']) and _is_const(x['t2'])),

    #2
    Rule(parse_str('BCH(prod(t1, M), prod(t2, N))'),
      ['M', 'N', 't1', 't2'],
      parse_str('prod(exp(prod(t1, M)), exp(prod(t2, N)), exp(prod(nsign(t1), M)), exp(prod(nsign(t2), N)))'),
      lambda x : _is_const(x['t1']) and _is_const(x['t2'])
    ),

    #12
    Rule(parse_str('exp( prod(?a, BOD(M, qb1)) )'),
        ['?a','M', 'qb1'],
        #exp(𝑖(𝜋/2)𝑎†𝑎)exp(𝑖(𝛼 (𝑎† + 𝑎)) ⊗ 𝜎𝑦)exp(−𝑖(𝜋/2)𝑎†𝑎)exp(𝑖(𝛼 (𝑎† + 𝑎)) ⊗ 𝜎𝑥 )
        #R(-pi/2)sdg()h()CD(j alpha)h()s()R(pi/2)h()CD(j alpha)h()
        parse_str(f'prod(qgate(R({-math.pi/2}), M), qgate(sdg, qb1), qgate(h, qb1), qgate(CD(?alphai), qb1, M), qgate(h, qb1), qgate(s, qb1), qgate(R({math.pi/2}), M), qgate(h, qb1), qgate(CD(?alphai), qb1, M), qgate(h, qb1))'),
        #parse_str('prod( qgate(h), qgate(CD(-1j)), qgate(h), qgate(R(pi/2)), qgate(h), qgate(sdg), qgate(CD(j)), qgate(s), qgate(h),  qgate(R(-pi/2)))'),
        lambda x: (__is_mode(x['M']) and
            _is_const(x['?a'])
            and (complex(x['?a'].name)/ (2j) == complex.conjugate(complex(x['?a'].name)/ (2j)))
        ),
        {'?alpha':(lambda x,_: complex(x['?a'].name)/ (2j)), '?alphai':(lambda x,_: complex(x['?a'].name)/ (2))}),
    #13
    Rule(parse_str('exp( prod(?a, BOD(dagger(M), qb1)) )'),
        ['?a','M', 'qb1'],
        #parse_str( 'prod( qgate(h), qgate(CD(i)), qgate(h), qgate(R(pi/2)), qgate(h), qgate(sdg), qgate(CD(i)), qgate(s), qgate(h),  qgate(R(-pi/2)))'  ),
        parse_str(f'prod(qgate(R({-math.pi/2}), M), qgate(sdg, qb1), qgate(h, qb1), qgate(CD(?alphai), qb1, M), qgate(h, qb1), qgate(s, qb1), qgate(R({math.pi/2}), qb1, M), qgate(h, qb1), qgate(CD(nsign(?alphai)), qb1, M), qgate(h, qb1))'),
        lambda x: (__is_mode(x['M']) and
            _is_const(x['?a'])
            and (complex(x['?a'].name)/ (2j) == complex.conjugate(complex(x['?a'].name)/ (2j)))
        ),
        {'?alpha':(lambda x,_: complex(x['?a'].name)/ (2j)), '?alphai':(lambda x,_: complex(x['?a'].name)/ (2))}),
]
