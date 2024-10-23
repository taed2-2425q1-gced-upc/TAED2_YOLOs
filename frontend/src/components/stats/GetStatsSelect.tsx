interface StatsCheckboxProps {
  onChange: (checked: boolean) => void
}

export const StatsCheckbox = ({onChange}: StatsCheckboxProps) => {
  return (
    <div className="flex items-center mb-2">
      <input
        className="appearance-none w-4 h-4 border border-gray-300 rounded-sm bg-gray-100 checked:bg-blue-600 checked:border-transparent focus:outline-none cursor-pointer"
        type="checkbox"
        name="stats"
        id="stats"
        onChange={(e) => onChange(e.target.checked)}
      />
      <label className="pl-3 text-sm text-gray-400" htmlFor="stats">
        Include stats
      </label>
    </div>
  )
}
