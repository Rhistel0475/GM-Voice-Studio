const PrepPanel = ({ title, children, className = "" }) => (
  <section className={`panel-ornate prep-panel ${className}`}>
    <div className="panel-head">
      <div className="plaque">{title}</div>
    </div>
    <div className="panel-body">{children}</div>
  </section>
);

export default PrepPanel;
