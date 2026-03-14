import QuickToolCard from "./QuickToolCard";

export default function QuickToolGrid({ tools, onToolClick }) {
  return (
    <div className="grid grid-cols-3 gap-3 min-h-[200px]">
      {tools.map((tool) => (
        <QuickToolCard key={tool.id} tool={tool} onClick={onToolClick} />
      ))}
    </div>
  );
}
