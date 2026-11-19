# Tools: design notes

Why each tool behaves as it does. The code says what it does; these notes keep the decisions behind
it, the alternatives that lost, and what was measured. One section per tool that has a ruling worth
keeping. The general rule they all share: **each tool follows the SQL operation it is named after**,
and that is how its handling of nulls is decided.

## join

**Three output ports instead of a `how` with many options.** One pairing gives `joined`, `left_only`
and `right_only`. A right join is the same join with the inputs swapped, an outer join reads all
three ports, and an anti join reads `left_only`, so none of those is an option. Two spellings of the
same result would be one more thing to get wrong. The unpaired rows become an edge a flow can send to
a report or an assertion, instead of a second join someone has to remember to write. `how: left`
was added because a left join is the join a flow asks for most, and building it from two steps was
the commonest piece of boilerplate.

**Fan-out is kept and shown.** A key that appears twice on the right turns one left row into two, and
an amount on that row is repeated, not split. Picking one match would need a rule the configuration
does not give. Keeping the key unique is the caller's job (remove_duplicates or assert_rows before
the join). The repetition stays visible so a reader sees it before trusting a total.

**A null key never pairs.** This follows SQL's `=`: a missing key is missing information, and pairing
two of them would invent a relationship. On dirty data it would also pair every null-keyed row on one
side with every null-keyed row on the other, turning a small gap into a large wrong number.
remove_duplicates decides the other way because it follows `DISTINCT`, where two nulls are one group.
Both are right for the operation they follow.

**Row accounting is about rows, not counts.** With fan-out, `joined.height + left_only.height ==
left.height` is false. The promise is that every row of a side is either on `joined` at least once or
on its own unpaired port exactly once, never both and never neither (under `how: inner`).

**Row order is fixed.** `joined` comes in left order, with a left row's copies in right order; each
unpaired port keeps its own side's order. This costs polars' `maintain_order`, and the parallelism it
gives up. It is paid so a flow that writes a file gets a diff that does not churn between runs. It is
not a promise to rely on for correctness: a step that needs an order still needs a sort.

**The unpaired ports are anti joins.** They are not a left join filtered on a null right-hand column,
because a paired row can hold a null in an ordinary column and would be miscounted as unpaired. Only
the other side's key columns are read.

**Clashing names are refused from the schema.** When the suffixed name of a right-hand column is
also taken, the step is refused before any row moves, naming `suffix` as the fix. Left to polars, it
fails later with a duplicate-column error that names neither the setting nor the fix.

**Key types are checked by asking polars.** Two empty one-column frames are joined to see whether
the types compare, so the check accepts exactly what a join accepts (Int32 with Int64 yes, Int64 with
Float64 no). A key column typed Null, as an empty branch gives, takes the other side's type so
`joined` keeps its usual columns; none of its rows pair.
